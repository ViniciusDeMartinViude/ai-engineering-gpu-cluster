import argparse, json, os
from pathlib import Path
import torch

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['smoke','train'], required=True)
    ap.add_argument('--model', default='yolo26n.pt')
    ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--imgsz',type=int,default=512)
    ap.add_argument('--batch',type=int,default=4)
    args=ap.parse_args()
    print('CUDA available:',torch.cuda.is_available(),flush=True)
    if not torch.cuda.is_available(): raise SystemExit('CUDA GPU not available')
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    if args.mode=='smoke':
        x=torch.rand((1024,1024),device='cuda'); print('CUDA tensor result:',float((x@x).mean()),flush=True)
    else:
        from ultralytics import YOLO
        from pathlib import Path
        yaml=Path('/dataset/data.yaml')
        if not yaml.is_file(): raise SystemExit('Missing /dataset/data.yaml')
        model=YOLO(args.model)
        model.train(data=str(yaml),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,
                    project='/output',name='train',exist_ok=True,device=0,workers=2,amp=False)
    print('peak_allocated_gib', round(torch.cuda.max_memory_allocated()/1024**3,3),flush=True)
    print('peak_reserved_gib', round(torch.cuda.max_memory_reserved()/1024**3,3),flush=True)
if __name__=='__main__': main()
