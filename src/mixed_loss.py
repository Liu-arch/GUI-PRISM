import torch
from torch.nn import functional as F
from runtime import input_for
KEYS=['head','lm','alignment','ranking','type','coordinate','presence','value','pointer']
def coordinate_loss(model,sample):
 # Click is coordinate-query conditioning, not class supervision.
 condition=dict(action_type='Click',point_2d=sample['point'])
 out=model.outputs(sample['prefix'],sample['plen'],gold=condition,forced_type='Click')
 bins=[round(float(z)*511/1000) for z in sample['point']];assert all(0<=v<512 for v in bins)
 loss=sum(F.cross_entropy(out[axis].float(),torch.tensor([b],device=out[axis].device)) for axis,b in zip(['x','y'],bins))
 zero=loss.new_zeros(());parts={k:zero for k in KEYS};parts['head']=loss;parts['coordinate']=loss
 return loss,parts
