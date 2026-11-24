"""Portable final-checkpoint initialization; no historical training directories."""
import torch,types
from torch import nn
from transformers import Qwen3VLForConditionalGeneration
from peft import LoraConfig,get_peft_model,get_peft_model_state_dict
import common
from latent_core import LatentPolicy
def adapter_cpu(model):return {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
ab=types.SimpleNamespace(adapter_cpu=adapter_cpu)
class FrozenC0:
 def __init__(self,c,out,rank):
  t=c['training'];self.c=c;self.device=torch.device('cuda',rank)
  model,info=Qwen3VLForConditionalGeneration.from_pretrained(t['model'],local_files_only=False,dtype=torch.bfloat16,device_map={'':rank},attn_implementation=t.get('attention','sdpa'),output_loading_info=True)
  assert not any(info.get(k,[]) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
  modules=[n for n,m in model.named_modules() if isinstance(m,nn.Linear) and n.startswith('model.language_model.layers.')]
  assert len(modules)==252
  model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=t['lora_r'],lora_alpha=t['lora_alpha'],target_modules=modules,lora_dropout=t['lora_dropout'],bias='none',use_rslora=t['lora_use_rslora']))
  for n,p in model.named_parameters():
   if p.requires_grad:
    assert 'lora_' in n and 'visual' not in n;p.data=p.data.float()
  self.policy=LatentPolicy(model,8,42);self.policy.requires_grad_(False);self.policy.eval()
  self.p=common.processor(t);self.tokenizer=self.p.tokenizer
  self.embedding=model.get_input_embeddings();self.output_weight=model.get_output_embeddings().weight
  self.bos=self.tokenizer.encode('<answer>',add_special_tokens=False)[-1];self.eos=self.tokenizer.eos_token_id
