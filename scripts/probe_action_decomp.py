#!/usr/bin/env python3
"""Decompose the offline rollout: is the problem the policy action, the WM, or the reward/reference?

Upper bound: roll out with the *reference* actions a_ref = (q_ref - default_q - bias)/scale
(the action the sim would need to follow the reference). If that scores ~0.09 / ~0.3 while the
policy conditioned on the recorded token scores 0.014 / 20.6, then WM+reward+references are
fine and the broken link is "token -> policy action".
"""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diff_vla_wbc.adapter import AdapterConfig, MotionGoalAdapter
from diff_vla_wbc.data import MotionBatchProvider, RecordedTokenSource
from diff_vla_wbc.isaac_joint import load_isaac_bundle
from diff_vla_wbc.rollout import (compute_joint_loss, imagine_with_tokens,
                                  _call_robot_bodies, _flat_history, _call_observation)
from diff_vla_wbc.train import JointTrainerConfig

dev = torch.device("cuda")
files = sorted(Path("data/npz/ha_tokens").glob("*.npz"))
bundle = load_isaac_bundle(Path("trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_013500.pt"),
                           device=dev, npz=files, packed_motion=Path("runs/isaac_motion/ha_tokens/packed.pt"),
                           packed_dir=None, obs_mode=None, hist=10, token_dim=64)
lib = bundle.library
ad = MotionGoalAdapter(AdapterConfig(token_dim=64, token_horizon=40, hidden=256,
                                     transformer_layers=2, transformer_heads=4)).to(dev)
rec = RecordedTokenSource(Path("data/npz/ha_tokens"), lib, token_horizon=40, token_dim=64, device=dev)
prov = MotionBatchProvider(bundle, lib, batch_size=2, horizon=40, token_source=None,
                           token_dim=64, token_horizon=40, seed=12345, hand_dir=None)
b = prov.sample()
H = 8
cfg = JointTrainerConfig(horizon=H)
motion = bundle.motion

def ref_action(frames):          # (B,) -> (B,29)
    rows = motion.ref[frames.long()].float()
    q = rows[..., 13:13+29]
    bias = b.xi[..., 1:1+29]
    return (q - bundle.default_joint_pos - bias) / bundle.action_scale

def rollout_actions(actions):
    state, xi = b.start_state, b.xi
    a_prev = b.prev_action
    history = b.history.clone()
    states, rewards = [state], []
    for t in range(H):
        ref = b.references[t]
        act = actions[:, t]
        nxt = bundle.step(state, act, xi)
        bodies = _call_robot_bodies(bundle, nxt)
        terms = bundle.reward(nxt, act, a_prev, ref, bodies)
        obs = _call_observation(bundle, nxt, act, xi, ref)
        history = torch.cat((history[1:], obs.unsqueeze(0)), 0)
        rewards.append(terms["total"]); states.append(nxt)
        a_prev = act; state = nxt
    return torch.stack(states), torch.stack(rewards)

# 参考动作序列：frames 后移 t 帧
fr = b.frames.clone()
# 第 t 步的目标是 f+t+1 帧 -> 用该帧的参考关节算动作
acts_ref = torch.stack([ref_action(fr + t + 1) for t in range(H)], dim=1)
st_ref, rw_ref = rollout_actions(acts_ref)
print(f"[ref-actions t+1 index] reward={float(rw_ref.mean()):+.5f}")

tok = rec({"frames": b.frames})
for name, t in [("labels", tok)]:
    with torch.no_grad():
        ro = imagine_with_tokens(bundle, tokens=t, state=b.start_state, obs_hist=b.obs_hist,
                                 history=b.history, xi=b.xi, references=b.references,
                                 prev_action=b.prev_action, horizon=H, obs_noise=0.0)
        _, terms = compute_joint_loss(ro, teacher_states=b.teacher_states, gamma=cfg.gamma, weights=cfg.weights)
    print(f"[policy/{name}] reward={float(ro.rewards.mean()):+.5f} state={float(terms['state']):.3f}")

# 状态误差对比（vs teacher_states）
for tag, st in [("ref-actions", st_ref[1:]), ("policy", ro.states[1:])]:
    errs = [float(((st[t]-b.teacher_states[:, t])**2)[..., :36].mean()) for t in range(min(H, st.shape[0]))]
    print(f"  {tag} 位置/关节 MSE per step: " + " ".join(f"{e:.3f}" for e in errs))

# 单步动作对比：策略 vs 参考（t=0）
obs_flat = _flat_history(bundle, b.history)
with torch.no_grad():
    a_pol = bundle.act(obs_flat, tok[:, 0], ref=b.references[0])
a_ref0 = acts_ref[:, 0]
print(f"\n[action t=0] |a_pol - a_ref|: mean={float((a_pol-a_ref0).abs().mean()):.4f} "
      f"max={float((a_pol-a_ref0).abs().max()):.4f}")
print(f"[action t=0] |a_pol|={float(a_pol.abs().mean()):.4f}  |a_ref|={float(a_ref0.abs().mean()):.4f}")
print(f"[action t=0] prev_action |a|={float(b.prev_action.abs().mean()):.4f}  "
      f"|a_pol-prev|={float((a_pol-b.prev_action).abs().mean()):.4f}")

print("\n[scale sweep] 参考动作 × α 后的 reward / 末步位置关节误差")
base = acts_ref
for alpha in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
    st, rw = rollout_actions(base * alpha)
    err = float(((st[-1] - b.teacher_states[:, H-1]) ** 2)[..., :36].mean())
    print(f"  alpha={alpha:>5}: reward={float(rw.mean()):+.5f}  state_last={err:.4f}")
print(f"\n[bundle] action_scale range=[{float(bundle.action_scale.min()):.5f}, {float(bundle.action_scale.max()):.5f}]")
print(f"[ref] |a_ref| mean={float(acts_ref.abs().mean()):.3f} max={float(acts_ref.abs().max()):.3f}")

print("\n[per-joint] 用 a_ref * action_scale（= 不再除 action_scale，即 a = Δq）")
st, rw = rollout_actions(acts_ref * bundle.action_scale)
err = float(((st[-1] - b.teacher_states[:, H-1]) ** 2)[..., :36].mean())
print(f"  reward={float(rw.mean()):+.5f}  state_last={err:.4f}")
print("[action_scale 唯一值]", sorted({round(float(v), 5) for v in bundle.action_scale.reshape(-1)}))
