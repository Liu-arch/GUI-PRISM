import json
from pathlib import Path
from PIL import Image
import torch
from latent_core import insert_slots
from structured_head import TEXT_TYPES
import common
from grounding import prompts

def input_for(enc,row,mode):
    with Image.open(row['image']) as im:
        image=im.convert('RGB')
        if mode=='trajectory':
            rendered=common.prefix(enc.p,{'user':row.get('user',row.get('instruction'))},'A')
        else:
            defs=json.loads((Path(__file__).parent.parent/'configs/prompts.json').read_text())
            system,user=prompts(mode,'A',{'image_size':image.size,'instruction':row.get('instruction',row.get('user'))},defs)
            messages=[{'role':'system','content':system},{'role':'user','content':[{'type':'image'},{'type':'text','text':user}]}]
            rendered=enc.p.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inp=enc.p(text=[rendered],images=[image],padding=False,truncation=False,return_tensors='pt')
    inp.pop('token_type_ids',None)
    if inp['input_ids'].shape[1]+8>enc.c['training']['max_sequence_tokens']:raise ValueError('Input exceeds token budget')
    return {k:v.to(enc.device) for k,v in inp.items()}

@torch.no_grad()
def text_value(enc,inputs,kind):
 """Only an unconstrained value string uses the preserved pretrained LM."""
 from transformers import StoppingCriteria,StoppingCriteriaList
 assert kind in TEXT_TYPES
 prefix='<answer>'+json.dumps(dict(action_type=kind),ensure_ascii=False,separators=(',',':'))[:-1]+',"value":"'
 prompt=insert_slots(inputs,inputs['input_ids'].shape[1],8,enc.tokenizer.pad_token_id)
 ids=enc.tokenizer.encode(prefix,add_special_tokens=False)
 prompt['input_ids']=torch.cat([prompt['input_ids'],prompt['input_ids'].new_tensor([ids])],1)
 prompt['attention_mask']=torch.ones_like(prompt['input_ids'])
 start=prompt['input_ids'].shape[1]
 class StopString(StoppingCriteria):
  def __call__(self,input_ids,scores,**kwargs):
   text=enc.tokenizer.decode(input_ids[0,start:],skip_special_tokens=True,clean_up_tokenization_spaces=False)
   try:return isinstance(json.JSONDecoder().raw_decode('"'+text)[0],str)
   except (json.JSONDecodeError,ValueError):return False
 enc.policy.slot_start=inputs['input_ids'].shape[1];enc.policy.prefill_only=True
 try:
  result=enc.policy.model.get_base_model().generate(**prompt,do_sample=False,max_new_tokens=1024,use_cache=True,stopping_criteria=StoppingCriteriaList([StopString()]))
 finally:enc.policy.slot_start=None;enc.policy.prefill_only=False
 tokens=result[0,start:].tolist()
 raw=enc.tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
 try:
  value,end=json.JSONDecoder().raw_decode('"'+raw)
  assert isinstance(value,str)
  return value,dict(tokens=len(tokens),raw_value_output=raw,parameter_error=None)
 except (ValueError,AssertionError):
  return None,dict(tokens=len(tokens),raw_value_output=raw,parameter_error='value_string_not_completed')
