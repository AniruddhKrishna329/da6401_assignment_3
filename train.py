"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

from weakref import ref
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
import wandb
from functools import partial
from evaluate import load as load_metric
from model import Transformer,make_src_mask,make_tgt_mask
from dataset import Multi30kDataset
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self,vocab_size,pad_idx,smoothing=0.1):
        super().__init__()
        self.pad=pad_idx
        self.eps=smoothing
        self.v=vocab_size

    def forward(self,logits,target):
        log_p=torch.log_softmax(logits,dim=-1)
        smooth=self.eps/(self.v-2)
        loss=-log_p.sum(-1)*smooth
        one_hot=torch.zeros_like(log_p).scatter_(1,target.unsqueeze(1),(1-self.eps-smooth*(self.v-2)))
        loss=loss-(one_hot*log_p).sum(-1)
        mask=(target!=self.pad)
        return loss[mask].mean()


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(data_iter,model,loss_fn,optimizer=None,scheduler=None,epoch_num=0,is_train=True,device='cpu'):
    model.train() if is_train else model.eval()
    total,n=0,0
    ctx=torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        pbar=tqdm(data_iter,desc=f"{'Train' if is_train else 'Val'} Epoch {epoch_num}")
        for src,tgt in pbar:
            src,tgt=src.to(device),tgt.to(device)
            src_mask=make_src_mask(src)
            tgt_in=tgt[:,:-1]
            tgt_out=tgt[:,1:]
            tgt_mask=make_tgt_mask(tgt_in)
            logits=model(src,tgt_in,src_mask,tgt_mask)
            loss=loss_fn(logits.reshape(-1,logits.size(-1)),tgt_out.reshape(-1))
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                if scheduler: scheduler.step()
            total+=loss.item();n+=1
            pbar.set_postfix({'loss':f'{total/n:.3f}'})
    avg=total/n
    wandb.log({'epoch':epoch_num,('train' if is_train else 'val')+'_loss':avg})
    print(f"{'Train' if is_train else 'Val'} Epoch {epoch_num} Loss: {avg:.3f}")
    return avg


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(model,src,src_mask,max_len,start_symbol,end_symbol,device='cpu'):
    src,src_mask=src.to(device),src_mask.to(device)
    mem=model.encode(src,src_mask)
    ys=torch.tensor([[start_symbol]],device=device)
    for _ in range(max_len):
        tm=make_tgt_mask(ys).to(device)
        out=model.decode(mem,src_mask,ys,tm)
        nxt=out[:,-1,:].argmax(-1).item()
        ys=torch.cat([ys,torch.tensor([[nxt]],device=device)],dim=1)
        if nxt==end_symbol: break
    return ys

# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(model,test_dataloader,tgt_vocab,device='cpu',max_len=100):
    model.eval()
    bleu=load_metric('bleu')
    sos,eos=tgt_vocab['<sos>'],tgt_vocab['<eos>']
    pad=tgt_vocab['<pad>']
    idx2tok={v:k for k,v in tgt_vocab.items()}
    preds,refs=[],[]
    with torch.no_grad():
        for src,tgt in test_dataloader:
            src=src.to(device)
            sm=make_src_mask(src)
            for i in range(src.size(0)):
                out=greedy_decode(model,src[i:i+1],sm[i:i+1],max_len,sos,eos,device)
                pred=[idx2tok.get(t,'<unk>') for t in out[0,1:].tolist() if t not in (eos,pad)]
                ref=[idx2tok.get(t,'<unk>') for t in tgt[i,1:].tolist() if t not in (eos,pad)]
                preds.append(' '.join(pred))
                refs.append([' '.join(ref)])
    score=bleu.compute(predictions=preds,references=refs)
    return score['bleu']*100

# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(model,optimizer,scheduler,epoch,path='checkpoint.pt'):
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'scheduler_state_dict':scheduler.state_dict(),
        'model_config':model.cfg
    },path)

def load_checkpoint(path,model,optimizer=None,scheduler=None):
    ck=torch.load(path,map_location='cpu')
    model.load_state_dict(ck['model_state_dict'])
    if optimizer: optimizer.load_state_dict(ck['optimizer_state_dict'])
    if scheduler: scheduler.load_state_dict(ck['scheduler_state_dict'])
    return ck['epoch']

def collate_fn(batch,pad_idx=1):
    src,tgt=zip(*batch)
    ml_s=max(len(s) for s in src)
    ml_t=max(len(t) for t in tgt)
    ps=[s+[pad_idx]*(ml_s-len(s)) for s in src]
    pt=[t+[pad_idx]*(ml_t-len(t)) for t in tgt]
    return torch.tensor(ps),torch.tensor(pt)

# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment():
    
    cfg={
        'd_model':256,'N':4,'num_heads':8,'d_ff':512,
        'dropout':0.2,'warmup':4000,'epochs':30,'batch':128,'lr':2.0
    }
    wandb.init(project='da6401-a3',config=cfg)

    from dataset import Multi30kDataset
    from  lr_scheduler import NoamScheduler

    train_ds=Multi30kDataset('train')
    val_ds=Multi30kDataset('validation')
    test_ds=Multi30kDataset('test')

    # share vocab across splits
    val_ds.vocab_de=test_ds.vocab_de=train_ds.vocab_de
    val_ds.vocab_en=test_ds.vocab_en=train_ds.vocab_en
    val_ds.processed=val_ds.process_data()
    test_ds.processed=test_ds.process_data()

    pad=1
    from functools import partial
    cf=partial(collate_fn,pad_idx=pad)
    train_dl=DataLoader(train_ds,batch_size=cfg['batch'],shuffle=True,collate_fn=cf)
    val_dl=DataLoader(val_ds,batch_size=cfg['batch'],collate_fn=cf)
    test_dl=DataLoader(test_ds,batch_size=1,collate_fn=cf)

    device='cuda' if torch.cuda.is_available() else 'cpu'
    sv,tv=len(train_ds.vocab_de),len(train_ds.vocab_en)

    model=Transformer(sv,tv,cfg['d_model'],cfg['N'],cfg['num_heads'],cfg['d_ff'],cfg['dropout'])
    model.cfg={'src_vocab_size':sv,'tgt_vocab_size':tv,'d_model':cfg['d_model'],
               'N':cfg['N'],'num_heads':cfg['num_heads'],'d_ff':cfg['d_ff'],'dropout':cfg['dropout']}
    model.to(device)

    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=(0.9,0.98),eps=1e-9)
    sch=NoamScheduler(opt,cfg['d_model'],cfg['warmup'])
    print(f"LR: {opt.param_groups[0]['lr']:.6f}")
    loss_fn=LabelSmoothingLoss(tv,pad,0.1)

    best_val=float('inf')
    for ep in range(cfg['epochs']):
        run_epoch(train_dl,model,loss_fn,opt,sch,ep,True,device)
        vl=run_epoch(val_dl,model,loss_fn,None,None,ep,False,device)
        if vl<best_val:
            best_val=vl
            save_checkpoint(model,opt,sch,ep,'best.pt')

    load_checkpoint('best.pt',model)
    bleu=evaluate_bleu(model,test_dl,train_ds.vocab_en,device)
    wandb.log({'test_bleu':bleu})
    print(f'BLEU: {bleu:.2f}')

if __name__ == "__main__":
    run_training_experiment()
