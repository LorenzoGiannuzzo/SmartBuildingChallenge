"""Break down world-model reward errors by component on held-out episodes."""
import torch, numpy as np, json
from pathlib import Path
from sbwm.wm_data import load_dir, Normalizer, HIST
from sbwm.world_model import WorldModel, Rollout
torch.set_num_threads(2)
ck=torch.load('wm_runs/v1/world_model.pt',weights_only=False)
m=WorldModel(); m.load_state_dict(ck['model']); m.eval(); n=Normalizer.from_state_dict(ck['norm'])
eps=load_dir(Path('wm_data/episodes_test'))
def comps(ep,t,temp,power):
    rm=ep.reward_meta; w=rm['productivity_deviation_ref']
    dev=torch.clamp(torch.maximum(temp-ep.csp[t],ep.hsp[t]-temp),min=0)
    sf=torch.where(dev<=w,dev/w,1+((dev-w)/w)**2); occ=ep.occ[t]; tot=occ.sum(-1)
    act=(300*(1-sf)*occ).sum(-1); reg=torch.where(tot>0,act/(300*tot).clamp_min(1e-9)-1,0.0)
    k=ep.schema.num_ahus; e=power[...,0]*80000*k; g=power[...,1]*205000*k; c=ep.coef[t]
    ne=(e*c[...,0]+g*c[...,1])/(rm['max_electricity_rate']*c[...,0]+rm['max_natural_gas_rate']*c[...,1])
    nc=(e*c[...,2]+g*c[...,3])/(rm['max_electricity_rate']*c[...,2]+rm['max_natural_gas_rate']*c[...,3])
    return torch.stack([0.6*reg,-0.2*ne,-0.2*nc],-1)
rng=np.random.default_rng(1); H=144
P=[];T=[];PW=[];TW=[]
with torch.no_grad():
    for ep in eps:
        t0=torch.as_tensor(rng.integers(HIST+2,ep.length-H-2,8)); ro=Rollout(m,n,ep); s=ro.init_state(t0)
        for i in range(H):
            t=t0+i; s,out,nt=ro.step(s,t,ep.zact[t],ep.aact[t],ep.hact[t])
            P.append(comps(ep,t,nt,out['power'])); T.append(comps(ep,t,ep.temp[t],ep.power[t])); PW.append(out['power']); TW.append(ep.power[t])
P=torch.cat(P);T=torch.cat(T);PW=torch.cat(PW);TW=torch.cat(TW)
print('component sums pred', P.sum(0).numpy().round(1), 'true', T.sum(0).numpy().round(1))
print('component corr', [round(float(np.corrcoef(P[:,i],T[:,i])[0,1]),3) for i in range(3)])
print('power aggregate rel err elec,gas', ((PW-TW).abs().sum(0)/TW.abs().sum(0)).numpy().round(3), 'bias', (PW.sum(0)/TW.sum(0)).numpy().round(3))
# comfort big errors: share of true comfort loss from large deviations
