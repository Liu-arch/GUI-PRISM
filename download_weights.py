import argparse,hashlib
from huggingface_hub import hf_hub_download

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='weights');a=p.parse_args()
    file=hf_hub_download('L1007555652/GUI-PRISM','GUI-PRISM.pt',local_dir=a.output)
    h=hashlib.sha256()
    with open(file,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!='7eca6a2a8ce9bd293c9973ab5b4a00141af683ea88e1295ab67ea5bc97f38e78':raise RuntimeError('Checkpoint checksum mismatch')
    print(file)
if __name__=='__main__':main()
