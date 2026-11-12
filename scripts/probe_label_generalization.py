#!/usr/bin/env python3
"""**标签泛化：留出一整个任务族 vs 留出运动（族见过）**（2026-11-07）。

为什么：`labels_v3` 的 38,400 条标签覆盖**全部 6 个任务族**
（double_desk 23.8% / vision_navi 19.1% / pp_box 17.0% / open_door 14.9% /
boxing 13.0% / sit_sofa 12.1%），而**现有留出段 100% 是 HOI_double_desk**
（因为"按 frame 分 10 段"里的 frame 是**运动库行号**，前 2/10 段正好落在库首的
double_desk 上）。所以：

* 历史报出的 `held-out cos 0.397/0.376` = **留出一整个任务族**的保守数，不是乐观数；
* 部署三任务（open_door / sit_sofa / vision_navi）在标签库里都有，但闭环用的是
  **没见过的运动** ⇒ 真正要对齐的口径是"族见过、运动没见过"。

本脚本用同一套特征/同一套 mask（只在非零标签帧上算 cos）给出这两组数。

用法::

    python3 scripts/probe_label_generalization.py --mode family   # 留一族
    python3 scripts/probe_label_generalization.py --mode motion   # 留运动（族见过）
"""

import argparse
import numpy as np, glob, torch, sys
sys.path.insert(0, '.')
sys.path.insert(0, 'scripts')
from train_distill_v3 import state64_from_state71


def _load():
    PK=torch.load('runs/isaac_motion/ha_cmd/packed.pt',map_location='cpu',weights_only=False)
    names=list(PK['names']); starts=np.asarray(PK['starts']); tot=int(np.asarray(PK['lengths']).sum())
    buf={}
    for f in sorted(glob.glob('runs/labels_v3/dump_*.npz')):
        z=np.load(f)
        for k in ("token_chunk","obs_deploy930","g_tok_pert","frame","state"):
            buf.setdefault(k,[]).append(z[k])
    D={k:np.concatenate(v,0) for k,v in buf.items()}
    N=len(D['frame'])
    X=np.concatenate([D['token_chunk'].astype(np.float32).reshape(N,-1),
                      D['obs_deploy930'].astype(np.float32),
                      state64_from_state71(D['state'].astype(np.float32))],1)
    G=torch.as_tensor(D['g_tok_pert'].astype(np.float32))
    nrm=G.norm(dim=-1); info=nrm>1e-9; Yu=G/nrm.clamp_min(1e-12)[...,None]
    mi=np.searchsorted(starts,np.clip(D['frame'].astype(np.int64),0,tot-1),side='right')-1
    fam=np.array([names[i].split('__')[0] for i in mi])
    return X, Yu, info, mi, fam, N, D['g_tok_pert'].shape[1]


def _fit_eval(X, Yu, info, te, tr, T, dev):
    Xt=torch.as_tensor(X,device=dev); Gm=torch.as_tensor(Yu,device=dev)
    mu=Xt[tr].mean(0,keepdim=True); sd=Xt[tr].std(0,keepdim=True).clamp_min(1e-8)
    A=((Xt[tr]-mu)/sd); B=((Xt[te]-mu)/sd)
    Gg=A.T@A+1e-2*max(1,len(tr))*torch.eye(A.shape[1],device=dev)
    W=torch.linalg.solve(Gg,A.T@Gm[tr].reshape(len(tr),-1))
    P=(B@W).reshape(len(te),T,64)
    c=(P*Gm[te]).sum(-1)/(P.norm(dim=-1).clamp_min(1e-9)*Gm[te].norm(dim=-1).clamp_min(1e-9))
    return float(c[info[te]].mean()), float(c.mean())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("family","motion"),default="family")
    ap.add_argument("--device",default="cuda:0"); a=ap.parse_args()
    X,Yu,info,mi,fam,N,T=_load(); dev=torch.device(a.device)
    print(f"[data] N={N} dim={X.shape[1]} 运动={len(set(mi.tolist()))} 族={len(set(fam.tolist()))}")
    rows=[]
    if a.mode=="family":
        for f in sorted(set(fam.tolist())):
            te=np.where(fam==f)[0]; tr=np.where(fam!=f)[0]
            if len(te)<200: continue
            ci,ca=_fit_eval(X,Yu,info,te,tr,T,dev); rows.append((f,len(te),ci,ca))
            print(f"  留族 {f:22s} n_te={len(te):5d}  cos_info={ci:+.3f}  cos_all={ca:+.3f}")
    else:
        rng=np.random.default_rng(0)
        for f in ("HSI_open_door","HSI_sit_sofa","HSI_vision_navi","HOI_double_desk"):
            m=np.where(fam==f)[0]; mids=np.unique(mi[m]); rng.shuffle(mids)
            te_m=set(mids[:max(1,int(0.3*len(mids)))].tolist())
            te=m[np.isin(mi[m],list(te_m))]; tr=np.setdiff1d(np.arange(N),te)
            ci,ca=_fit_eval(X,Yu,info,te,tr,T,dev); rows.append((f,len(te),ci,ca))
            print(f"  留运动 {f:22s} {len(te_m):2d}/{len(mids):2d} 条运动 n_te={len(te):5d}  "
                  f"cos_info={ci:+.3f}  cos_all={ca:+.3f}")
    w=sum(n*c for _,n,c,_ in rows)/sum(n for _,n,_,_ in rows)
    print(f"\n加权 cos_info = {w:.3f}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
