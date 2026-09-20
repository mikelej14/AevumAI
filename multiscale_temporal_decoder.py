"""0.9.34 multi-scale temporal decoder.

Keeps native 1 ms route-event timing, but avoids asking one Gaussian width to resolve
both tiny and broad timing differences.  Each recurrent edge is compared at several
scales and by its inter-event interval (ISI) shape.  Fine scales preserve tiny timing
separations; coarse scales retain jitter tolerance.  No neural-physics changes.
"""
from collections import defaultdict
from math import exp, sqrt
from real_run_memory import frame_similarity, diff_similarity

SCALES=(0.50, 1.00, 2.00, 4.00)
SCALE_WEIGHTS=(0.34, 0.29, 0.23, 0.14)

def _edges(ep):
    d=defaultdict(list)
    for s,tgt,t in ep.get("route_events",()): d[(s,tgt)].append(float(t))
    for v in d.values(): v.sort()
    return d

def _nearest_pair_kernel(a,b,sigma,window):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    used=[False]*len(b); q=0.0; n=0
    for x in a:
        bi=-1; bd=1e300
        for i,y in enumerate(b):
            if not used[i]:
                dt=abs(x-y)
                if dt < bd: bi,bd=i,dt
        if bi >= 0 and bd <= window:
            used[bi]=True; n+=1
            q += exp(-0.5*(bd/sigma)**2)
    union=len(a)+len(b)-n
    return q/union if union else 1.0

def _isi_similarity(a,b,sigma=1.0):
    # Translation-invariant local rhythm: timing gaps survive a uniform onset shift.
    if len(a)<2 and len(b)<2: return 1.0
    if len(a)<2 or len(b)<2: return 0.0
    da=[a[i+1]-a[i] for i in range(len(a)-1)]
    db=[b[i+1]-b[i] for i in range(len(b)-1)]
    return _nearest_pair_kernel(da,db,sigma,4.0*sigma)

def temporal_signature_similarity(a,b):
    A=_edges(a); B=_edges(b); keys=set(A)|set(B)
    if not keys: return {"combined":1.0,"multiscale":1.0,"isi":1.0,"fine":1.0}
    den=0.0; ms=isi=fine=0.0
    for k in keys:
        x=A.get(k,()); y=B.get(k,()); w=max(len(x),len(y),1); den+=w
        vals=[_nearest_pair_kernel(x,y,s,4.0*s) for s in SCALES]
        m=sum(v*wgt for v,wgt in zip(vals,SCALE_WEIGHTS))
        ms += w*m; fine += w*vals[0]; isi += w*_isi_similarity(x,y,0.75)
    ms/=den; fine/=den; isi/=den
    # Rhythm is supporting evidence, not a replacement for absolute event time.
    combined=0.82*ms+0.18*isi
    return {"combined":combined,"multiscale":ms,"isi":isi,"fine":fine}

def score(cue,e):
    s=frame_similarity(cue["frames"],e["frames"])
    d=diff_similarity(cue["frames"],e["frames"])
    t=temporal_signature_similarity(cue,e)
    # Temporal evidence dominates this resolution-focused branch.
    total=.18*s+.12*d+.70*t["combined"]
    return total,s,d,t

class MultiScaleTemporalMemory:
    def __init__(self): self.families={}
    def store(self,label,e,provenance=None): self.families.setdefault(label,[]).append((e,provenance))
    def recall(self,cue):
        rows=[]
        for lab,items in self.families.items():
            candidates=[]
            for i,(e,_) in enumerate(items):
                sc,s,d,t=score(cue,e)
                candidates.append((sc,i,s,d,t))
            sc,i,s,d,t=max(candidates,key=lambda z:z[0])
            rows.append({"label":lab,"score":sc,"matched_exemplar":i,"state":s,"diff":d,
                         "temporal":t["combined"],"fine":t["fine"],"isi":t["isi"]})
        return sorted(rows,key=lambda r:r["score"],reverse=True)
