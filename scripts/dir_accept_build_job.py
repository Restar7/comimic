"""方向验收：把 <FT> 相对 <BASE> 的实际动作位移方向拿去真机测(±δ)。用法: build_job_dir.py BASE FT OUT"""
import sys, glob, json, torch, numpy as np
sys.path.insert(0,'.')
from diff_vla_wbc.isaac_joint import load_isaac_bundle, FrameRef
from diff_vla_wbc.reward_cfg import apply_reward_cfg
from diff_vla_wbc.sonic_mjlab import ha_to_std, std_to_ha, action_scale_sonic
import joint_contract as JC
dev=torch.device('cuda:0')
# SONIC 序动作尺度：从 joint_contract 的置换现算（不再依赖 /tmp 或手工 npy）
ss=action_scale_sonic().numpy().astype('float32'); ss_t=torch.tensor(ss)
ha=torch.tensor(ha_to_std(),dtype=torch.long); sh=torch.tensor(std_to_ha(),dtype=torch.long)

# ---------- 1) FT 方向（SONIC 单位 / SONIC 序） ----------
def load(run):
    rows=[json.loads(l) for l in open(f'runs/closedloop_r3/{run}/dump/steps.jsonl') if l.strip()]
    return (np.array([r['applied'] for r in rows]), np.array([r['frame'] for r in rows]))
import sys as _s
Ab,Fb=load(_s.argv[1]); Af,Ff=load(_s.argv[2])
lut={int(f):i for i,f in enumerate(Ff)}
pairs=[(i,lut[int(f)]) for i,f in enumerate(Fb) if int(f) in lut][:400]
dq=np.stack([Af[j]-Ab[i] for i,j in pairs])                 # 绝对关节目标差(rad, SONIC 序)
d_a = torch.tensor((dq/ss[None,:]).astype('float32')).mean(0)                   # → 归一化动作单位
print("[p0] 匹配帧 %d，FT 动作位移 RMS = %.4f (SONIC 单位) ≈ %.4f rad RMS"%(
    len(pairs), float(d_a.pow(2).mean().sqrt()), float((d_a*ss_t).pow(2).mean().sqrt())))
d_hat = d_a/max(float(d_a.pow(2).mean().sqrt()),1e-9)        # RMS 归一化

# ---------- 2) 语料窗口 ----------
npz=sorted(glob.glob('data/npz/ha_tokens/*.npz'))
b=load_isaac_bundle('runs/S1_handover/WBC_A24000_wmvel_cmd.pt', device=dev, npz=npz,
    packed_motion='runs/isaac_motion/ha_cmd_exec/packed.pt', obs_mode=None, hist=None, token_dim=64)
apply_reward_cfg(b, json.load(open('runs/joint_v8/args.json')))
ck=torch.load('runs/a_prime_sonic_exc/wm_gain2.pt',map_location='cpu',weights_only=False)
b.wm.load_state_dict(ck['wm'], strict=True); b.wm.requires_grad_(False); b.wm.eval()
d=torch.load('runs/a_prime_sonic_leash/DoubleDesk_off0.pt',map_location='cpu',weights_only=False)
S=d['states'].float(); A=d['actions'].float(); F=d['frames'].long()
stride,K,N=64,20,48
g=torch.Generator().manual_seed(0)
starts=torch.randperm(len(S)-(K-1)*stride,generator=g)[:N]
rows=starts[:,None]+torch.arange(K)[None,:]*stride
s0=S[starts].to(dev); a0=A[rows].to(dev); fr=F[rows].to(dev); xi=b.default_xi(N,dev)
spec=torch.load('runs/S1_handover/WBC_A24000_wmvel_cmd.pt',map_location='cpu',weights_only=False)['robot_spec']
ha_scale=torch.tensor(np.asarray(spec['action_scale'],dtype='float32'))
def Jof(a):
  with torch.enable_grad():
    s=s0; tot=torch.zeros((),device=dev)
    for t in range(K):
        s=b.step(s, a[:,t], xi, step_idx=t+1)
        a_ha=(a[:,t].index_select(-1,ha.to(dev))*ss_t.to(dev)[ha.to(dev)])/ha_scale.to(dev)
        tot=tot+b.reward(s,a_ha,a_ha,FrameRef(fr[:,t]))['total']
  return tot.sum()/N
delta=0.15
ap=(a0+delta*d_hat.to(dev)).detach(); am=(a0-delta*d_hat.to(dev)).detach()
J0=Jof(a0); Jp=Jof(ap); Jm=Jof(am)
print("[p0] WM 沿 FT 方向：J(a0)=%.4f J(+)=%.4f J(-)=%.4f ΔJ=%+.4f  ⇒ WM 认为 %s 更好"%(
    float(J0),float(Jp),float(Jm),float(Jp-Jm), "+ (FT 方向)" if float(Jp)>float(Jm) else "− (FT 反方向)"))
def to_ha(a): return (a*ss_t.to(a.device)).index_select(-1,ha.to(a.device))/ha_scale.to(a.device)
def st_ha(st):
    o=st.clone(); o[:,7:36]=st[:,7:36].index_select(-1,sh.to(st.device)); o[:,42:71]=st[:,42:71].index_select(-1,sh.to(st.device)); return o
OUT=_s.argv[3]
torch.save({"states_ha":st_ha(s0.cpu()),"a0":a0.cpu(),"acts_plus":to_ha(ap).cpu(),"acts_minus":to_ha(am).cpu(),
  "prev":to_ha(a0[:,0].cpu()),"frames":fr.cpu(),"horizon":K,"delta":delta,
  "J_WM_plus":torch.tensor(float(Jp)),"J_WM_minus":torch.tensor(float(Jm)),
  "joint_names":list(JC.HA_V31_29),
  "meta":{"dir":"FT realized action delta (from closed-loop dumps)"}}, OUT)
print(f"[dir] job → {OUT}")
