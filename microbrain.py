
from dataclasses import dataclass,asdict
from collections import defaultdict
import random,math,json,csv

@dataclass
class Config:
 rows:int=8; cols:int=16; seed:int=7; rest:float=-65.; reset:float=-68.; threshold:float=-54.; tau:float=22.; refractory:int=3
 sensory:int=24; output_start:int=120; output_count:int=8; route_ttl:int=16; history:int=10; plasticity:bool=False
 @property
 def n(self): return self.rows*self.cols

@dataclass
class Neuron:
 v:float=-65.; refractory:int=0; spikes:int=0; last:int=-999999

@dataclass(frozen=True)
class Route:
 id:int; origin:int; path:tuple; born:int

class Brain:
 def __init__(self,c=None):
  self.c=c or Config(); self.r=random.Random(self.c.seed); self.n=[Neuron(self.c.rest) for _ in range(self.c.n)]
  self.edges=self._hex(); self.w={}; self.delay={}; self.pending=defaultdict(list); self.events=[]; self.t=0; self.rid=1; self.trace_transmissions=False; self.transmissions=[]
  self.capture_events=True; self.last_fired=tuple(); self.transmission_recorder=None
  self.decoder_lexicon={}; self.decoder_route_fingerprints={}; self.decoder_route_labels={}; self.neural_pattern_bank={}; self.neural_prototype_bank={}; self.neural_prototype_routes={}; self.neural_token_prototypes={}; self.neural_prototype_postings={}; self.neural_memories=[]; self.neural_memory_index={}; self.neural_documents=[]; self.memory_next_id=0; self.document_next_id=0; self._wire()

 def _hex(self):
  R,C=self.c.rows,self.c.cols; a=[[] for _ in range(R*C)]
  for r in range(R):
   for col in range(C):
    q=[(r,col-1),(r,col+1),(r-1,col-1),(r-1,col),(r+1,col-1),(r+1,col)] if r%2==0 else [(r,col-1),(r,col+1),(r-1,col),(r-1,col+1),(r+1,col),(r+1,col+1)]
    a[r*C+col]=[rr*C+cc for rr,cc in q if 0<=rr<R and 0<=cc<C]
  return a

 def _wire(self):
  # Baseline keeps the original ~2/3 excitatory / ~1/3 inhibitory hex graph.
  # Excitation is deliberately sub-threshold alone: propagation is driven by
  # temporal summation and converging routes rather than every spike avalanching.
  for i,ns in enumerate(self.edges):
   ordered=ns[:]; self.r.shuffle(ordered); ec=round(len(ordered)*2/3)
   for k,j in enumerate(ordered):
    self.w[i,j]=self.r.uniform(.72,1.02) if k<ec else -self.r.uniform(.22,.42)
    self.delay[i,j]=self.r.randint(1,4)

 def set_synapse(self,i,j,w,delay=None):
  i,j=int(i),int(j)
  if i==j: raise ValueError("self-connections are disabled")
  if not (0<=i<self.c.n and 0<=j<self.c.n): raise ValueError("neuron out of range")
  if j not in self.edges[i]: self.edges[i].append(j)
  self.w[i,j]=float(w)
  self.delay[i,j]=max(1,int(delay if delay is not None else self.delay.get((i,j),2)))

 def remove_synapse(self,i,j):
  i,j=int(i),int(j)
  if j in self.edges[i]: self.edges[i].remove(j)
  self.w.pop((i,j),None); self.delay.pop((i,j),None)

 def inject(self,i,amp=24.,at=None,origin=None):
  at=self.t if at is None else int(at); rid=self.rid; self.rid+=1
  self.pending[at].append((i,float(amp),Route(rid,i if origin is None else origin,(i,),at)))
  return rid

 def symbol(self,s,start=None):
  start=self.t+1 if start is None else start; rr=random.Random(1000003+int(s)); cells=rr.sample(range(self.c.sensory),6)
  for k,(i,g) in enumerate(zip(cells,[0,2,5,9,14,20])): self.inject(i,24,start+g+(k*(s%3))%3,i)
  return cells

 def _independence(self,rs):
  v=[]
  for x in range(len(rs)):
   for y in range(x+1,len(rs)):
    A=set(rs[x].path[:-1]);B=set(rs[y].path[:-1]);U=A|B;v.append(1-len(A&B)/len(U) if U else 1.)
  return sum(v)/len(v) if v else 0.

 def _stdp(self,i,j):
  if not self.c.plasticity:return
  dt=self.n[j].last-self.n[i].last
  if not (-80<=dt<=80) or dt==0:return
  dw=.010*math.exp(-dt/20) if dt>0 else -.012*math.exp(dt/20)
  s=1 if self.w[i,j]>=0 else -1;m=abs(self.w[i,j])
  m=max(.05,min(2.5,m+(dw if s>0 else -dw)));self.w[i,j]=s*m

 def step(self):
  cur=[0.]*self.c.n; routes=defaultdict(list)
  for i,a,r in self.pending.pop(self.t,[]):cur[i]+=a;routes[i].append(r)
  fired=[]
  for i,x in enumerate(self.n):
   if x.refractory:x.refractory-=1;x.v=self.c.reset;continue
   x.v+=(self.c.rest-x.v)/self.c.tau+cur[i]
   if x.v>=self.c.threshold:fired.append(i)
  self.last_fired=tuple(fired)
  for i in fired:
   x=self.n[i];x.v=self.c.reset;x.refractory=self.c.refractory;x.spikes+=1;x.last=self.t
   rs=routes[i];ind=self._independence(rs); origins=sorted({r.origin for r in rs})
   if self.capture_events:
    self.events.append(dict(t=self.t,type="spike",neuron=i,route_count=len(rs),route_ids=[r.id for r in rs],origins=origins,independence=ind))
    if len(rs)>=2:self.events.append(dict(t=self.t,type="convergence",neuron=i,route_count=len(rs),route_ids=[r.id for r in rs],origins=origins,independence=ind))
   if not rs:rs=[Route(self.rid,i,(i,),self.t)];self.rid+=1
   for j in self.edges[i]:
    self._stdp(i,j)
    for r in rs:
     if len(r.path)>=self.c.route_ttl or j in r.path:continue
     nr=Route(r.id,r.origin,r.path+(j,),r.born)
     arrival=self.t+self.delay[i,j]
     self.pending[arrival].append((j,10*self.w[i,j],nr))
     if self.transmission_recorder is not None:
      self.transmission_recorder.add_transmission(self.t,i,j,r.origin,r.born)
     elif self.trace_transmissions:
      self.transmissions.append(dict(send_t=self.t,arrival_t=arrival,source=i,destination=j,
       weight=self.w[i,j],delay=self.delay[i,j],amplitude=10*self.w[i,j],
       route_id=r.id,origin=r.origin,route=list(nr.path),born=r.born))
  self.t+=1

 def run(self,ms):
  for _ in range(int(ms)):self.step()
  return self.analyze()
 
 def reset_activity(self,clear_events=False):
  # Reset transient electrical state only. Topology/weights/delays are untouched.
  self.pending.clear()
  for x in self.n:
   x.v=self.c.rest;x.refractory=0
  if clear_events:self.events.clear();self.transmissions.clear()

 def reward(self,targets,amount=1.,window=60):
  active={e["neuron"] for e in self.events if e["type"]=="spike" and e["t"]>=self.t-window}
  changed=0
  for (i,j),w in list(self.w.items()):
   if w>0 and i in active and (j in active or j in targets):self.w[i,j]=min(2.5,w+.015*amount);changed+=1
  self.events.append(dict(t=self.t,type="reward",targets=list(targets),changed=changed))

 def readout(self,window=50):
  score={i:0. for i in range(self.c.output_start,self.c.output_start+self.c.output_count)}
  for e in self.events:
   if e.get("t",-1)<self.t-window or e.get("neuron") not in score:continue
   score[e["neuron"]]+=1 if e["type"]=="spike" else e["route_count"]*(1+e["independence"])
  return max(score,key=score.get),score

 def analyze(self):
  sp=[e for e in self.events if e["type"]=="spike"];cv=[e for e in self.events if e["type"]=="convergence"]; per=[0]*self.c.n
  for e in sp:per[e["neuron"]]+=1
  E=sum(v>0 for v in self.w.values());I=len(self.w)-E
  return dict(time_ms=self.t,neurons=self.c.n,synapses=len(self.w),excitatory=E,inhibitory=I,excitatory_fraction=E/len(self.w),spikes=len(sp),active_neurons=sum(x>0 for x in per),convergences=len(cv),mean_independence=sum(e["independence"] for e in cv)/len(cv) if cv else 0,top_firing=sorted(enumerate(per),key=lambda z:z[1],reverse=True)[:12])

 def neuron(self,i):
  return dict(id=i,v=self.n[i].v,spikes=self.n[i].spikes,last=self.n[i].last,
   outgoing=[dict(to=j,weight=self.w[i,j],delay=self.delay[i,j]) for j in self.edges[i]],
   incoming=[dict(fr=j,weight=self.w[j,i],delay=self.delay[j,i]) for j in range(self.c.n) if (j,i) in self.w])

 def export(self,p="microbrain"):
  json.dump(self.events,open(p+"_events.json","w"),indent=2)
  json.dump(dict(config=asdict(self.c),synapses=[dict(pre=i,post=j,weight=w,delay=self.delay[i,j]) for (i,j),w in self.w.items()]),open(p+"_network.json","w"),indent=2)
  json.dump(self.analyze(),open(p+"_analysis.json","w"),indent=2)
  with open(p+"_spikes.csv","w",newline="") as f:
   q=csv.writer(f);q.writerow(["time","neuron","routes","origins","independence"])
   for e in self.events:
    if e["type"]=="spike":q.writerow([e["t"],e["neuron"],e["route_count"],"|".join(map(str,e["origins"])),e["independence"]])

MicroBrain=Brain
