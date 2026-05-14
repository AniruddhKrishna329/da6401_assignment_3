"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

from spacy import tokens
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(Q,K,V,mask=None):
    dk=Q.size(-1)
    scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(dk)
    if mask is not None:
        scores=scores.masked_fill(mask,-1e9)
    w=F.softmax(scores,dim=-1)
    return torch.matmul(w,V),w


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src,pad_idx=1):
    return (src==pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt,pad_idx=1):
    b,t=tgt.shape
    pad_mask=(tgt==pad_idx).unsqueeze(1).unsqueeze(2)
    causal=torch.triu(torch.ones(t,t,device=tgt.device),diagonal=1).bool()
    return pad_mask|causal


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self,d_model,num_heads,dropout=0.1):
        super().__init__()
        assert d_model%num_heads==0
        self.h=num_heads
        self.dk=d_model//num_heads
        self.wq=nn.Linear(d_model,d_model)
        self.wk=nn.Linear(d_model,d_model)
        self.wv=nn.Linear(d_model,d_model)
        self.wo=nn.Linear(d_model,d_model)
        self.drop=nn.Dropout(dropout)
    
    def forward(self,query,key,value,mask=None):
        b=query.size(0)
        Q=self.wq(query).view(b,-1,self.h,self.dk).transpose(1,2)
        K=self.wk(key).view(b,-1,self.h,self.dk).transpose(1,2)
        V=self.wv(value).view(b,-1,self.h,self.dk).transpose(1,2)
        x,_=scaled_dot_product_attention(Q,K,V,mask)
        x=x.transpose(1,2).contiguous().view(b,-1,self.h*self.dk)
        return self.wo(x)


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super().__init__()
        self.drop=nn.Dropout(dropout)
        pe=torch.zeros(max_len,d_model)
        pos=torch.arange(0,max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000)/d_model))
        pe[:,0::2]=torch.sin(pos*div)
        pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer('pe',pe.unsqueeze(0))

    def forward(self,x):
        return self.drop(x+self.pe[:,:x.size(1)])


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.l1=nn.Linear(d_model,d_ff)
        self.l2=nn.Linear(d_ff,d_model)
        self.drop=nn.Dropout(dropout)

    def forward(self,x):
        return self.l2(self.drop(F.relu(self.l1(x))))



# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super().__init__()
        self.attn=MultiHeadAttention(d_model,num_heads,dropout)
        self.ff=PositionwiseFeedForward(d_model,d_ff,dropout)
        self.n1=nn.LayerNorm(d_model)
        self.n2=nn.LayerNorm(d_model)
        self.drop=nn.Dropout(dropout)

    def forward(self,x,src_mask):
        x=self.n1(x+self.drop(self.attn(x,x,x,src_mask)))
        return self.n2(x+self.drop(self.ff(x)))


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super().__init__()
        self.attn1=MultiHeadAttention(d_model,num_heads,dropout)
        self.attn2=MultiHeadAttention(d_model,num_heads,dropout)
        self.ff=PositionwiseFeedForward(d_model,d_ff,dropout)
        self.n1=nn.LayerNorm(d_model)
        self.n2=nn.LayerNorm(d_model)
        self.n3=nn.LayerNorm(d_model)
        self.drop=nn.Dropout(dropout)

    def forward(self,x,memory,src_mask,tgt_mask):
        x=self.n1(x+self.drop(self.attn1(x,x,x,tgt_mask)))
        x=self.n2(x+self.drop(self.attn2(x,memory,memory,src_mask)))
        return self.n3(x+self.drop(self.ff(x)))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self,layer,N):
        super().__init__()
        self.layers=nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm=nn.LayerNorm(layer.n1.normalized_shape[0])

    def forward(self,x,mask):
        for l in self.layers: x=l(x,mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self,layer,N):
        super().__init__()
        self.layers=nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm=nn.LayerNorm(layer.n1.normalized_shape[0])

    def forward(self,x,memory,src_mask,tgt_mask):
        for l in self.layers: x=l(x,memory,src_mask,tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(self,src_vocab_size=18669,tgt_vocab_size=9797,d_model=256,N=4,num_heads=8,d_ff=512,dropout=0.2,checkpoint_path='best.pt'):
        super().__init__()
        import spacy
        from datasets import load_dataset
        import subprocess
        subprocess.run(["python","-m","spacy","download","de_core_news_sm"])
        subprocess.run(["python","-m","spacy","download","en_core_web_sm"])
        self.nlp_de=spacy.load("de_core_news_sm")
        self.nlp_en=spacy.load("en_core_web_sm")
        self.nlp_de=spacy.load("de_core_news_sm")
        self.nlp_en=spacy.load("en_core_web_sm")
        data=load_dataset("bentrevett/multi30k",split="train")
        specials=['<unk>','<pad>','<sos>','<eos>']
        vde={t:i for i,t in enumerate(specials)}
        ven={t:i for i,t in enumerate(specials)}
        for item in data:
            for t in [x.text.lower() for x in self.nlp_de.tokenizer(item['de'])]:
                if t not in vde: vde[t]=len(vde)
            for t in [x.text.lower() for x in self.nlp_en.tokenizer(item['en'])]:
                if t not in ven: ven[t]=len(ven)
        self.vocab_de=vde
        self.vocab_en=ven
        self.idx2en={v:k for k,v in ven.items()}
        self.src_emb=nn.Embedding(src_vocab_size,d_model)
        self.tgt_emb=nn.Embedding(tgt_vocab_size,d_model)
        self.pe=PositionalEncoding(d_model,dropout)
        enc_layer=EncoderLayer(d_model,num_heads,d_ff,dropout)
        dec_layer=DecoderLayer(d_model,num_heads,d_ff,dropout)
        self.encoder=Encoder(enc_layer,N)
        self.decoder=Decoder(dec_layer,N)
        self.proj=nn.Linear(d_model,tgt_vocab_size)
        self.scale=math.sqrt(d_model)
        self.cfg={'src_vocab_size':src_vocab_size,'tgt_vocab_size':tgt_vocab_size,
                'd_model':d_model,'N':N,'num_heads':num_heads,'d_ff':d_ff,'dropout':dropout}
        if not os.path.exists(checkpoint_path):
            gdown.download(id="1dzzw_8xLEmI6i51Jpd0jY5IFIIPXCq5d",output=checkpoint_path,quiet=False)
        ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        self.load_state_dict(ck['model_state_dict'])

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
    
        return self.encoder(self.pe(self.src_emb(src)*self.scale),src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        return self.proj(self.decoder(self.pe(self.tgt_emb(tgt)*self.scale),memory,src_mask,tgt_mask))

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        return self.decode(self.encode(src,src_mask),src_mask,tgt,tgt_mask)


    def infer(self,src_sentence,max_len=50):
        self.eval()
        sos,eos,pad=2,3,1
        tokens=[sos]+[self.vocab_de.get(t.text.lower(),0) for t in self.nlp_de.tokenizer(src_sentence)]+[eos]
        src=torch.tensor(tokens).unsqueeze(0)
        src_mask=make_src_mask(src)
        with torch.no_grad():
            mem=self.encode(src,src_mask)
            tgt=torch.tensor([[sos]])
            for _ in range(max_len):
                tm=make_tgt_mask(tgt)
                out=self.decode(mem,src_mask,tgt,tm)
                nxt=out[:,-1,:].argmax(-1).item()
                if nxt==eos: break
                tgt=torch.cat([tgt,torch.tensor([[nxt]])],dim=1)
        return ' '.join(self.idx2en.get(i,'<unk>') for i in tgt[0,1:].tolist())
