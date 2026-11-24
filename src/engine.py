import json
import torch
from v22_model import Model,state,restore
from runtime import input_for
from mixed_loss import coordinate_loss
from v22_inference import predict_v22
import common

class Engine:
    def __init__(self,config,rank,stage='joint',checkpoint=None):
        self.config=config
        self.model=Model(config,None,rank,stage)
        if checkpoint:restore(self.model,torch.load(checkpoint,map_location='cpu',weights_only=True))
    def prepare(self,row):
        mode=row.get('mode','trajectory')
        inp=input_for(self.model.enc,row,mode)
        sample={'prefix':inp,'plen':inp['input_ids'].shape[1],'spec':row.get('spec',{})}
        if row.get('coordinate_only',False):
            sample['point']=row['point_2d'];sample['coordinate_only']=True
        else:
            sample['action']=row['action']
            ids,_,_=common.target_encoding(self.model.enc.tokenizer,{'action':row['action']},'A')
            if sample['plen']+8+len(ids)>self.config['training']['max_sequence_tokens']:raise ValueError('Training sequence exceeds token budget')
            sample['labels']=torch.tensor(ids,device=self.model.enc.device)
        return sample
    def loss(self,sample):
        if sample.get('coordinate_only'):return coordinate_loss(self.model,sample)[0]
        return self.model(sample)[0]
    def optimizer(self):return self.model.optimizer()
    def state(self):return state(self.model)
    def predict(self,row,benchmark):
        self.model.eval()
        inp=input_for(self.model.enc,row,'trajectory' if benchmark=='action' else benchmark)
        return predict_v22(self.model,inp,benchmark)
