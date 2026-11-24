import json,time,torch
from runtime import text_value,TEXT_TYPES
@torch.no_grad()
def predict_v22(mod,inputs,benchmark=None):
 enc=mod.enc;assert not mod.training
 inputs={k:v.to(enc.device) for k,v in inputs.items()}
 plen=inputs['input_ids'].shape[1];torch.cuda.synchronize();start=time.time()
 with torch.autocast('cuda',dtype=torch.bfloat16):out=mod.outputs(inputs,plen)
 action=mod.action(out);torch.cuda.synchronize();elapsed=time.time()-start
 note=dict(tokens=0,raw_value_output=None,parameter_error=None)
 if action['action_type'] in TEXT_TYPES:action['value'],note=text_value(enc,inputs,action['action_type'])
 audit=dict(native_action_before_task_serialization=dict(action),query_condition_type=out['query_condition_type'],query_condition_x=out['query_condition_x'],coordinate_bins=512)
 if benchmark in ('osworldg','screenspot') and action['action_type']=='Click':action['value']='None'
 if benchmark=='osworldg' and action['action_type']=='Terminate':
  action['point_2d']=[-100,-100];action['value']='Requested target cannot be located in this screenshot.'
 raw='<answer>'+json.dumps(action,ensure_ascii=False,separators=(',',':'))+'</answer>';torch.cuda.synchronize()
 return dict(raw_response_text=raw,action=action,head_audit=audit,**note,generated_tokens=note['tokens'],
  at_token_limit=note['tokens']>=1024 and bool(note['parameter_error']),batch_latency_seconds=time.time()-start,
  encoder_seconds=elapsed,structured_head_seconds=0.,timing_note='Backbone and query head interleaved; reported jointly under encoder_seconds',
  prediction_mode='v22_structured_head',batch_size=1,latent_count=8,latent_injection_verified=True)
