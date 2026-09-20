
"""End-to-end episodic memory from actual MicroBrain runs."""
from collections import Counter
from microbrain import Brain,Config
from stimulus import inject_stimulus
from route_decoder import weighted_jaccard

def run_experience(text, tail_ms=45):
 b=Brain(Config(plasticity=False));b.trace_transmissions=True
 frames=[];anchors=[]
 for ch in text:
  start=b.t+1;anchors.append(start);inject_stimulus(b,ch,start=start)
  # calibrated glyph lasts 20 ms; retain recurrent aftermath too
  for _ in range(32):
   b.step()
   active=tuple(sorted(e["neuron"] for e in b.events if e["type"]=="spike" and e["t"]==b.t-1))
   frames.append((b.t-start,active))
 for _ in range(tail_ms):
  b.step();active=tuple(sorted(e["neuron"] for e in b.events if e["type"]=="spike" and e["t"]==b.t-1))
  frames.append((b.t-anchors[-1],active))
 routes=Counter((x["source"],x["destination"],(x["send_t"]-anchors[0])//10) for x in b.transmissions)
 return {"text":text,"frames":frames,"routes":routes,"transmissions":len(b.transmissions),"route_events":tuple((x["source"],x["destination"],x["send_t"]-anchors[0]) for x in b.transmissions),"causal_events":tuple((x["origin"],x["source"],x["destination"],x["send_t"]-anchors[0],x["born"]-anchors[0]) for x in b.transmissions)}

def j(a,b):
 A=set(a);B=set(b);return len(A&B)/len(A|B) if A or B else 1.
def frame_similarity(a,b):
 n=min(len(a),len(b))
 if not n:return 0.
 return sum(j(a[i][1],b[i][1]) for i in range(n))/n
def differential(frames):
 return [(tuple(sorted(set(b[1])-set(a[1]))),tuple(sorted(set(a[1])-set(b[1])))) for a,b in zip(frames,frames[1:])]
def diff_similarity(a,b):
 A=differential(a);B=differential(b);n=min(len(A),len(B))
 if not n:return 0.
 return sum((j(A[i][0],B[i][0])+j(A[i][1],B[i][1]))/2 for i in range(n))/n

class RealRunEpisodeMemory:
 def __init__(self):self.records=[]
 def store(self,episode,label):
  self.records.append({"id":len(self.records),"label":label,"episode":episode})
 def recall(self,cue):
  rows=[]
  for r in self.records:
   e=r["episode"];s=frame_similarity(cue["frames"],e["frames"]);d=diff_similarity(cue["frames"],e["frames"])
   p=weighted_jaccard(cue["routes"],e["routes"])
   score=.35*s+.25*d+.40*p
   rows.append({"id":r["id"],"label":r["label"],"score":score,"state":s,"diff":d,"route":p})
  return sorted(rows,key=lambda x:x["score"],reverse=True)
