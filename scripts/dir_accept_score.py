"""打分：真物理里 ±梯度方向 的动作串，哪一串的 20 步漂移更小？与 WM 的判断是否一致？"""
import sys, torch, numpy as np
sys.path.insert(0,'.')
from diff_vla_wbc.isaac_joint import canonical_to_flat
import sys as _s
R=torch.load(_s.argv[2],map_location='cpu',weights_only=False)
J=torch.load(_s.argv[1],map_location='cpu',weights_only=False)
pk=torch.load('runs/isaac_motion/ha_cmd_exec/packed.pt',map_location='cpu',weights_only=False)
ref=canonical_to_flat(torch.as_tensor(pk['ref']).float())      # 与 state 同布局
P=R['states_plus'].float(); M=R['states_minus'].float()          # (N,H,71) ha 序（root 块无碍）
S0=J['states_ha'].float(); F=J['frames'].long()                  # (N,71), (N,H)
N,H,_=P.shape
RF=ref[F]                                                        # (N,H,71)
p0=S0[:,:3]
def drift(X):
    d=(X[...,:3]-p0[:,None,:]) - (RF[...,:3]-RF[:,0:1,:3])
    return d.norm(dim=-1)                                        # (N,H)
dp, dm = drift(P), drift(M)
# 用整段的平均与末步两种口径
for name, f in (("整段均值", lambda x: x.mean(1)), ("末步", lambda x: x[:,-1])):
    a, b = f(dp), f(dm)
    diff = b - a                        # >0 ⇒ plus 臂漂移更小 ⇒ WM 的"+"方向是对的
    print("[score] %s: plus %.2f cm | minus %.2f cm | Δ(minus−plus)=%+.2f cm | FT方向 更好 %d/%d 窗"%(
        name, float(a.mean())*100, float(b.mean())*100, float(diff.mean())*100,
        int((diff>0).sum()), N))
d=dm[:,-1]-dp[:,-1]
print("\n[score] WM 说 FT方向 更好（ΔJ_WM=+0.217，构造上必然）")
print("[score] 真物理 FT方向 更好：%d/%d 窗 = %.0f%%"%(int((d>0).sum()), N, 100*float((d>0).float().mean())))
print("[score] 差值量级：均值 %+.2f cm，中位 %+.2f cm，|Δ| 中位 %.2f cm  （对照 WM 的 20 步误差 ≈0.89 cm）"%(
    float(d.mean())*100, float(d.median())*100, float(d.abs().median())*100))
print("[score] ⇒ 判定:", "✅ WM 的 20 步方向在真机上成立" if float((d>0).float().mean())>=0.65
      else ("❌ WM 的 20 步方向不成立（≈随机）" if float((d>0).float().mean())<=0.55 else "⚠ 弱信号，需更多窗口"))
