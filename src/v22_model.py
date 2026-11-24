"""C0 interface around the unchanged v22 role-gated Action-Query core."""
import os,json,re,math,hashlib,contextlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from encoder import FrozenC0,ab,common
from latent_core import insert_slots,alignment_loss
from structured_head import ACTIONS,TEXT_TYPES,MANDATORY_POINT,OPTIONAL_POINT
from qwen3_gui_agent.lm_action_query_policy import LMActionQueryPolicy
R=Path(__file__).resolve().parent

class V22Head(nn.Module):
 def __init__(self,c):
  super().__init__();h=c['head'];self.enums=c['enums'];d=h['hidden_dim']
  os.environ['LARA_ACTION_QUERY_CONTEXT_SOURCES']='reasoning,boundary,action,visual'
  self.core=LMActionQueryPolicy(input_dim=4096,action_type_count=len(ACTIONS),terminate_count=3,
   hidden_dim=d,num_heads=h['heads'],coord_bins=h['coord_bins'],layer_count=8,dropout=0,
   teacher_forcing_ratio=c['v22_teacher_forcing_ratio'],interaction_mode='independent',layer_mix_mode='shared',
   context_role_mode='role_gated',latent_role_layout='all_reasoning',router_mode='none',coordinate_prior_mode='none',
   role_projection_mode='independent_residual',residual_router_mode='action_to_visual',
   role_adapter_dim=512,action_adapter_dim=4096,residual_gate_init=-2.,
   pointer_action_indices=[i for i,k in enumerate(ACTIONS) if k in MANDATORY_POINT|OPTIONAL_POINT])
  self.action_slots=nn.Parameter(torch.randn(6,4096)*.02)
  self.presence=nn.Linear(d,2)
  self.values=nn.ModuleDict({k:nn.Linear(d,len(v)) for k,v in self.enums.items()})
 def memory(self,h,inp,plen,enc):
  ids=inp['input_ids'][0];image=(ids==enc.p.image_token_id).nonzero().flatten()
  grid=inp['image_grid_thw'];m=enc.p.image_processor.merge_size
  hh,ww=int(grid[0,1])//m,int(grid[0,2])//m
  assert len(image)==hh*ww and h.shape[1]==plen+14
  ctx=torch.cat([h[:,plen:plen+8],h[:,plen+7:plen+8],h[:,plen+8:plen+14]],1)
  roles=h.new_tensor([self.core.MEMORY_ROLES.index('reasoning')]*8+[self.core.MEMORY_ROLES.index('boundary')]+[self.core.MEMORY_ROLES.index('action')]*6,dtype=torch.long)[None]
  mem,mask,role=self.core.encode_memory(current_visual_tokens=h[:,image],current_visual_token_mask=torch.ones((1,len(image)),device=h.device,dtype=torch.bool),
   target_patch_grid_sizes=[(hh,ww)],context_states=ctx,context_mask=torch.ones(ctx.shape[:2],device=h.device,dtype=torch.bool),context_role_ids=roles,return_role_ids=True)
  return dict(memory=mem,memory_mask=mask,memory_role_ids=role),h[:,plen:plen+8]
 def query(self,layers,memory,role):
  q=self.core.mix_hidden_layers(layers,role=role)[:,-1]
  state=self.core.decode_state(q,**memory,query_role=role)
  return q,state

class Model(nn.Module):
 def __init__(self,c,out,rank,stage):
  super().__init__();self.c=c;self.stage=stage;self.enc=FrozenC0(c,out,rank);self.policy=self.enc.policy
  torch.manual_seed(c['seed']+99);self.head=V22Head(c).to(self.enc.device)
  self.set_stage(stage)
 def set_stage(self,stage):
  self.stage=stage
  for n,p in self.policy.named_parameters():
   p.requires_grad_(stage=='joint' and ('lora_' in n or n=='latent_embeddings'))
   if p.requires_grad:p.data=p.data.float()
  self.policy.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  self.policy.model.enable_input_require_grads();self.policy.model.config.use_cache=False
 def optimizer(self):
  groups=[dict(params=list(self.head.parameters()),name='head',lr=self.c['stages'][self.stage]['lr'])]
  if self.stage=='joint':
   groups += [dict(params=[p for n,p in self.policy.model.named_parameters() if p.requires_grad],name='lora',lr=self.c['lora_lr']),
              dict(params=[self.policy.latent_embeddings],name='latent',lr=self.c['latent_lr'])]
  return torch.optim.AdamW(groups,weight_decay=self.c['weight_decay'])
 def states(self,inp,plen,suffix=None,action_slots=False,multi=False,grad=True):
  full=insert_slots(inp,plen,8,self.enc.tokenizer.pad_token_id)
  if action_slots:tokens=[self.enc.tokenizer.pad_token_id]*6
  else:tokens=[] if suffix is None else suffix
  if tokens:
   ids=full['input_ids'];full['input_ids']=torch.cat([ids,ids.new_tensor([tokens])],1);full['attention_mask']=torch.ones_like(full['input_ids'])
  handle=None
  if action_slots:
   def replace(module,args,out):
    a=self.head.action_slots.to(out.dtype)[None]
    return torch.cat([out[:,:plen+8],a,out[:,plen+14:]],1)
   handle=self.enc.embedding.register_forward_hook(replace)
  assert self.policy.slot_start is None;self.policy.slot_start=plen
  try:
   with torch.set_grad_enabled(grad):
    x=self.policy.model.get_base_model().model(**full,use_cache=False,return_dict=True,output_hidden_states=multi)
    return x.last_hidden_state,(list(x.hidden_states[-8:]) if multi else None)
  finally:
   self.policy.slot_start=None
   if handle is not None:handle.remove()
 def query_text(self,kind='Click',axis='type',x=0):
  text,_=common.target(dict(action=dict(action_type=kind,value='None',point_2d=[0,0])),'A')
  m=re.search(r'"action_type"\s*:\s*"',text);assert m
  if axis=='type':return text[:m.end()]
  m=re.search(r'"point_2d"\s*:\s*\[',text);assert m
  return text[:m.end()]+(str(x) if axis=='y' else '')
 def query_forward(self,inp,plen,text):
  return self.states(inp,plen,self.enc.tokenizer.encode(text,add_special_tokens=False),multi=True,grad=self.training and self.stage=='joint')[1]
 def outputs(self,inp,plen,gold=None,forced_type=None):
  h,_=self.states(inp,plen,action_slots=True,grad=self.training)
  memory,latent=self.head.memory(h,inp,plen,self.enc)
  qt,st=self.head.query(self.query_forward(inp,plen,self.query_text()),memory,'type')
  typ,_=self.head.core.classify_action_type(st)
  typ,pointer,gates=self.head.core.fuse_action_type_residual(query_state=qt,**memory,action_type_logits=typ)
  predicted=typ.detach().argmax(-1)
  teacher=predicted.new_tensor([ACTIONS.index(gold['action_type'])]) if gold else None
  cond=self.head.core.training_condition(teacher,predicted)
  kind=forced_type if forced_type is not None else ACTIONS[int(cond[0])]
  qx,sx=self.head.query(self.query_forward(inp,plen,self.query_text(kind,'x')),memory,'x')
  x=self.head.core.x_head(sx)
  px=x.detach().argmax(-1)
  tx=px.new_tensor([round(gold['point_2d'][0]*511/1000)]) if gold and isinstance(gold['point_2d'],list) else None
  cx=self.head.core.training_condition(tx,px)
  qy,sy=self.head.query(self.query_forward(inp,plen,self.query_text(kind,'y',round(int(cx[0])*1000/511))),memory,'y')
  y=self.head.core.y_head(sy)
  x,y,gates=self.head.core.fuse_action_coordinate_residual(x_query_state=qx,y_query_state=qy,**memory,x_logits=x,y_logits=y)
  return dict(type=typ,x=x,y=y,presence=self.head.presence(st),values={k:v(st) for k,v in self.head.values.items()},pointer=pointer,latent=latent,
              query_condition_type=kind,query_condition_x=int(cx[0]),residual_gates=gates)
 def forward(self,sample,teacher=None):
  out=self.outputs(sample['prefix'],sample['plen'],sample['action'] if self.training else None)
  a=sample['action'];device=self.enc.device
  ce=lambda logits,i:F.cross_entropy(logits.float(),torch.tensor([i],device=device))
  typ=ce(out['type'],ACTIONS.index(a['action_type']))
  present=isinstance(a['point_2d'],list);pres=ce(out['presence'],int(present))
  coord=typ.new_zeros(());ranking=typ.new_zeros(())
  if present:
   bins=[round(float(z)*511/1000) for z in a['point_2d']]
   assert all(0<=v<512 for v in bins)
   coord=ce(out['x'],bins[0])+ce(out['y'],bins[1])
   peer=sample['spec'].get('peer_point')
   if peer:
    pb=[round(float(z)*511/1000) for z in peer]
    if bins!=pb:
     lx=F.log_softmax(out['x'].float(),-1)[0];ly=F.log_softmax(out['y'].float(),-1)[0]
     ranking=F.relu(1.-lx[bins[0]]-ly[bins[1]]+lx[pb[0]]+ly[pb[1]])
  val=typ.new_zeros(())
  if a['action_type'] in self.head.enums:val=ce(out['values'][a['action_type']],self.head.enums[a['action_type']].index(str(a['value'])))
  pointer=typ.new_zeros(())
  if out['pointer'] is not None:pointer=ce(out['pointer'],int(a['action_type'] in MANDATORY_POINT|OPTIONAL_POINT))
  head=typ+coord+.25*pres+val+pointer
  lm=typ.new_zeros(());align=typ.new_zeros(())
  if self.stage=='joint':
   labels=sample['labels']
   h,_=self.states(sample['prefix'],sample['plen'],labels.tolist(),grad=self.training)
   states=h[0,sample['plen']+7:-1];assert len(states)==len(labels)
   for j in range(0,len(labels),64):
    logits=self.policy.model.get_output_embeddings()(states[j:j+64])
    lm+=F.cross_entropy(logits.float(),labels[j:j+64],reduction='sum')/len(labels)
   align=alignment_loss(out['latent'][0],teacher) if teacher is not None else lm.new_zeros(())
  total=head+self.c['lm_weight']*lm+self.c['alignment_weight']*align+self.c['ranking_weight']*ranking
  return total,dict(head=head,lm=lm,alignment=align,ranking=ranking,type=typ,coordinate=coord,presence=pres,value=val,pointer=pointer)
 def action(self,out,forced_type=None):
  kind=forced_type or ACTIONS[int(out['type'][0].argmax())]
  point=[round(int(out[z][0].argmax())*1000/511,3) for z in ['x','y']]
  use=kind in MANDATORY_POINT or (kind in OPTIONAL_POINT and int(out['presence'][0].argmax())==1)
  value=None if kind in TEXT_TYPES else self.head.enums[kind][int(out['values'][kind][0].argmax())]
  return dict(action_type=kind,value=value,point_2d=point if use else 'None')
def state(mod):
 return dict(adapter=ab.adapter_cpu(mod.policy.model),latent=mod.policy.latent_embeddings.detach().cpu().clone(),head={k:v.detach().cpu().clone() for k,v in mod.head.state_dict().items()})
def restore(mod,p):
 from peft import set_peft_model_state_dict
 set_peft_model_state_dict(mod.policy.model,p['adapter'])
 with torch.no_grad():mod.policy.latent_embeddings.copy_(p['latent'])
 mod.head.load_state_dict(p['head'],strict=True)
