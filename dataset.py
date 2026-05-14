import spacy
from datasets import load_dataset

class Multi30kDataset:
    def __init__(self, split='train'):
        self.split=split
        self.data=load_dataset("bentrevett/multi30k", split=split)
        self.nlp_de=spacy.load("de_core_news_sm")
        self.nlp_en=spacy.load("en_core_web_sm")
        self.vocab_de,self.vocab_en=self.build_vocab()
        self.processed=self.process_data()
    def tokenize(self,text,lang='de'):
        nlp=self.nlp_de if lang=='de' else self.nlp_en
        return [t.text.lower() for t in nlp.tokenizer(text)]
    def build_vocab(self):
        specials=['<unk>','<pad>','<sos>','<eos>']
        vocab_de={t:i for i,t in enumerate(specials)}
        vocab_en={t:i for i,t in enumerate(specials)}
        for item in self.data:
            for t in self.tokenize(item['de'],'de'):
                if t not in vocab_de: vocab_de[t]=len(vocab_de)
            for t in self.tokenize(item['en'],'en'):
                if t not in vocab_en: vocab_en[t]=len(vocab_en)
        return vocab_de,vocab_en

    def process_data(self):
        unk,sos,eos=0,2,3
        out=[]
        for item in self.data:
            de=[sos]+[self.vocab_de.get(t,unk) for t in self.tokenize(item['de'],'de')]+[eos]
            en=[sos]+[self.vocab_en.get(t,unk) for t in self.tokenize(item['en'],'en')]+[eos]
            out.append((de,en))
        return out
    def __len__(self): return len(self.processed)
    def __getitem__(self,idx): return self.processed[idx]