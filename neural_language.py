
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import json, math, os

@dataclass
class Glyph:
    label: str
    trials: int
    stable_neurons: list
    stable_edges: list
    stable_convergences: list
    timing_bins: dict
    signature: str
    confidence: float

class NeuralLanguage:
    """Learns deterministic signatures from repeated MicroBrain event traces."""
    def __init__(self, path="neural_dictionary.json"):
        self.path = path
        self.glyphs = {}
        if os.path.exists(path):
            try:
                raw=json.load(open(path,encoding="utf8"))
                self.glyphs={k:Glyph(**v) for k,v in raw.get("glyphs",{}).items()}
            except Exception:
                self.glyphs={}

    def save(self):
        json.dump({"version":"LRNR-DICT-0.1",
                   "glyphs":{k:asdict(v) for k,v in self.glyphs.items()}},
                  open(self.path,"w",encoding="utf8"),indent=2)

    @staticmethod
    def _features(events, start_t, end_t):
        spikes=[e for e in events if e.get("type")=="spike" and start_t<=e.get("t",-1)<=end_t]
        conv=[e for e in events if e.get("type")=="convergence" and start_t<=e.get("t",-1)<=end_t]
        neurons=Counter(e["neuron"] for e in spikes)
        edges=Counter()
        # Route history is represented indirectly in current event format by route IDs;
        # stable transitions are inferred from consecutive spikes sharing a route ID.
        by_route=defaultdict(list)
        for e in spikes:
            for rid in e.get("route_ids",[]):
                by_route[rid].append((e["t"],e["neuron"]))
        for seq in by_route.values():
            seq=sorted(set(seq))
            for a,b in zip(seq,seq[1:]):
                if b[0]>=a[0]:
                    edges[(a[1],b[1])]+=1
        convergences=Counter(e["neuron"] for e in conv)
        timing=defaultdict(list)
        if spikes:
            t0=min(e["t"] for e in spikes)
            for e in spikes:
                timing[e["neuron"]].append(e["t"]-t0)
        return neurons,edges,convergences,timing

    def learn(self,label,trials,stability=0.60):
        """trials = [(events,start_t,end_t), ...]"""
        nseen=Counter(); eseen=Counter(); cseen=Counter(); times=defaultdict(list)
        for events,s,e in trials:
            ns,es,cs,ts=self._features(events,s,e)
            nseen.update(ns.keys()); eseen.update(es.keys()); cseen.update(cs.keys())
            for n,vals in ts.items(): times[n].extend(vals)
        total=max(1,len(trials))
        stable_n=sorted(n for n,c in nseen.items() if c/total>=stability)
        stable_e=sorted([list(e) for e,c in eseen.items() if c/total>=stability])
        stable_c=sorted(n for n,c in cseen.items() if c/total>=stability)
        timing_bins={}
        for n in stable_n:
            vals=times.get(n,[])
            if vals:
                timing_bins[str(n)]=[round(min(vals),1),round(sum(vals)/len(vals),1),round(max(vals),1)]
        confidence=(sum(nseen[n]/total for n in stable_n)/max(1,len(stable_n))) if stable_n else 0.0
        sig=self.ascii_signature(label,stable_n,stable_e,stable_c,timing_bins)
        g=Glyph(str(label),len(trials),stable_n,stable_e,stable_c,timing_bins,sig,round(min(1,confidence),4))
        self.glyphs[str(label)]=g; self.save(); return g

    def score(self,events,start_t,end_t,glyph):
        ns,es,cs,ts=self._features(events,start_t,end_t)
        obs_n=set(ns); obs_e=set(es); obs_c=set(cs)
        gn=set(glyph.stable_neurons); ge={tuple(x) for x in glyph.stable_edges}; gc=set(glyph.stable_convergences)
        def j(a,b):
            u=a|b
            return len(a&b)/len(u) if u else 0.0
        structural=.50*j(obs_n,gn)+.30*j(obs_e,ge)+.20*j(obs_c,gc)
        timing_scores=[]
        for n,bounds in glyph.timing_bins.items():
            n=int(n)
            if n in ts and ts[n]:
                mean=sum(ts[n])/len(ts[n]); lo,mid,hi=bounds
                span=max(4.0,hi-lo,abs(mid)*.25)
                timing_scores.append(max(0,1-abs(mean-mid)/span))
        timing=sum(timing_scores)/len(timing_scores) if timing_scores else 0
        return .75*structural+.25*timing

    def identify(self,events,start_t,end_t):
        ranked=sorted(((self.score(events,start_t,end_t,g),name)
                       for name,g in self.glyphs.items()),reverse=True)
        return ranked

    @staticmethod
    def ascii_signature(label,neurons,edges,conv,timing):
        lines=[f"GLYPH {label}",f"NODES: {' '.join('N'+str(n) for n in neurons)}",
               "PATHS:"]
        for a,b in edges[:24]:
            lines.append(f"  N{a:03d} -> N{b:03d}")
        if not edges: lines.append("  (no stable route yet)")
        lines.append("CONVERGENCE: "+(" ".join("N"+str(n) for n in conv) if conv else "(none)"))
        lines.append("TIMING:")
        for n,v in list(timing.items())[:24]:
            lines.append(f"  N{int(n):03d}: {v[0]}..{v[2]} ms  mean={v[1]}")
        return "\n".join(lines)
