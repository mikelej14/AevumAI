
"""Deterministic LRNR route-motif decoder. No model inference required."""
from collections import Counter,defaultdict
from statistics import median
import json

def route_tokens(transmissions, anchor_ms, start_rel=-18, end_rel=52, bin_ms=10):
    c=Counter()
    for x in transmissions:
        rel=x["send_t"]-anchor_ms
        if start_rel <= rel <= end_rel:
            c[(x["source"],x["destination"],rel//bin_ms)] += 1
    return c

def weighted_jaccard(a,b):
    keys=set(a)|set(b)
    if not keys:return 1.0
    den=sum(max(a.get(k,0),b.get(k,0)) for k in keys)
    return sum(min(a.get(k,0),b.get(k,0)) for k in keys)/den if den else 1.0

def learn_motif(samples, presence=0.75):
    """samples: iterable of Counter route-token observations."""
    samples=list(samples); seen=Counter(); vals=defaultdict(list)
    for s in samples:
        for k,v in s.items():seen[k]+=1;vals[k].append(v)
    need=max(1,int(len(samples)*presence + .999999))
    return {k:median(vals[k]) for k,n in seen.items() if n>=need}

class RouteDecoder:
    def __init__(self): self.motifs={}
    def add(self,label,motif):self.motifs[label]=dict(motif)
    def identify(self,tokens):
        scores={label:weighted_jaccard(tokens,m) for label,m in self.motifs.items()}
        if not scores:return None,{}
        return max(scores,key=scores.get),scores
    def save(self,path):
        obj={lab:[[list(k),v] for k,v in m.items()] for lab,m in self.motifs.items()}
        json.dump(obj,open(path,"w"),indent=2)
    @classmethod
    def load(cls,path):
        d=cls();obj=json.load(open(path))
        for lab,items in obj.items():d.motifs[lab]={tuple(k):v for k,v in items}
        return d
