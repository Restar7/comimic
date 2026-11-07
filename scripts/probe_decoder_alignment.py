#!/usr/bin/env python3
"""训练代理执行器 ↔ 部署 SONIC release decoder 的**数值对拍**（会话 01 核心实验）。

三个必须分开回答的问题（混在一起就会得出错误结论）：

E1 图级等价：``SonicDecoderTorch``（训练 BPTT 走的可微实现）与
   ``onnxruntime`` 加载的**真实部署 ONNX**，在同一份 994 输入上是否逐位一致？
E2 token 通道：同一个 obs 下，token 扰动 δ 引起的动作扰动在两条路上是否同向同幅？
   （如果 E1 通过，这一项必然通过；做它是为了给出"∂a/∂token 没有被代理歪掉"的直接证据）
E3 obs 组装：**真正会不一致的地方**。同一物理状态，换一种 994 拼装口径
   （关节序、last_action 错一帧、q 偏移、token 量化），绝对动作误差多大？
   **更重要的是** —— token 的 Jacobian 方向是否还一样？（绝对误差小 ≠ Jacobian 相同）
E4 Δtoken 落地：真实 checkpoint 的 Δ（正确特征序 vs 错误特征序）经过真实 decoder
   后，动作修正的方向 cos / 范数比各是多少？
E5 代理 == 部署：把录制里的 994 反拼成训练侧 930 观测，经 ``SonicExecutor``（训练执行器）
   与真实 ONNX 对拍 —— 验证"训练用的就是真解码器"这句话在**数据上**成立。

数据来源：``data/ha_raw/*/*/*.npz`` 的 ``decoder_obs`` / ``decoder_raw_action`` —— 这是
部署时真正喂给 ONNX 的东西（该 npz 还记录了当时用的 decoder 路径）。

用法::

    python3 scripts/probe_decoder_alignment.py \
        --out runs/harness_01_deploy/decoder_alignment.csv \
        --json runs/harness_01_deploy/decoder_alignment.json \
        --arms bptt1=runs/bptt1/vla_000050.pt gt7fix=runs/gt7fix/vla_final.pt
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_ROOT = Path(os.environ.get("COMIMIC_DATA_ROOT", str(ROOT)))

import torch  # noqa: E402

import joint_contract as JC  # noqa: E402
from diff_vla_wbc import delta_token_io as DIO  # noqa: E402
from diff_vla_wbc.sonic_decoder import (  # noqa: E402
    BLOCKS,
    HIST,
    DECODER_IN_DIM,
    SonicDecoderTorch,
)
from diff_vla_wbc.sonic_mjlab import (  # noqa: E402
    DEFAULT_DECODER_ONNX,
    SonicExecutor,
    ha_to_std,
    std_to_ha,
)

P_HA2STD = np.asarray(JC.permutation("ha_v31", "standard"), dtype=np.int64)
#: SONIC release 的默认关节角（**HA_V31/isaaclab 序**，抄自
#: `action_provider_sonic.py:742-772`；已在 `observation_config_sonic_release.yaml` 与
#: 录制 ``decoder_obs`` 上验证：obs[94:384] ≡ joint_pos_sonic − SONIC_DEFAULT_POS）
SONIC_DEFAULT_POS_HA = np.array([
    -0.312, -0.312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.669, 0.669, 0.2, 0.2,
    -0.363, -0.363, 0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=np.float32)

BLOCK_SLICES = {"q": (94, 384), "qd": (384, 674), "a_prev": (674, 964),
                "ang_vel": (64, 94), "gravity": (964, 994)}


# --------------------------------------------------------------------------- ORT
class OrtDecoder:
    def __init__(self, path: str | Path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None]
        out = [self.sess.run(None, {self.name: obs[i:i + 1].copy()})[0][0]
               for i in range(len(obs))]          # ONNX batch 维度静态为 1
        return np.stack(out, 0)


# --------------------------------------------------------------------------- 工具
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def _jac_token(fn, obs: np.ndarray, token: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """中心差分 ``∂a/∂token``（29×64）。``fn(obs) -> (29,)``，obs 前 64 维是 token。"""
    base = np.array(obs, dtype=np.float32, copy=True)
    J = np.zeros((29, 64), dtype=np.float64)
    for j in range(64):
        p = base.copy(); p[:64] = token; p[j] += eps
        m = base.copy(); m[:64] = token; m[j] -= eps
        J[:, j] = (fn(p[None])[0] - fn(m[None])[0]) / (2 * eps)
    return J


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, float), -1e9, 1e9)


# --------------------------------------------------------------------------- 数据
def load_frames(path: Path, n: int) -> dict:
    z = np.load(path, allow_pickle=True)
    T = int(z["num_frames"])
    idx = np.unique(np.linspace(5, T - 2, min(n, T - 7)).astype(int))
    out = {
        "path": str(path),
        "idx": idx,
        "obs994": np.asarray(z["decoder_obs"], np.float32)[idx],
        "raw_action": np.asarray(z["decoder_raw_action"], np.float32)[idx],
        "target_action": np.asarray(z["decoder_target_action"], np.float32)[idx],
        "q_hist": np.asarray(z["encoder_robot_joint_pos_hist"], np.float32)[idx],
        "qd_hist": np.asarray(z["encoder_robot_joint_vel_hist"], np.float32)[idx],
        "av_hist": np.asarray(z["decoder_ang_vel_hist"], np.float32)[idx],
        "g_hist": np.asarray(z["decoder_gravity_dir_hist"], np.float32)[idx],
        "la_hist": np.asarray(z["decoder_last_action_hist"], np.float32)[idx],
        "state64": np.asarray(z["vla_state"], np.float32)[idx],
        "latent": np.asarray(z["encoder_latent"], np.float32)[idx],
        "decoder_path_recorded": str(z["meta_decoder_path"]) if "meta_decoder_path" in z else "",
        "T": T,
    }
    # q 块自检：obs[94:384] 必须== q_hist − SONIC_DEFAULT（这就是 decoder 要的偏移口径）
    # obs[94:384] 就是 `robot_qpos − SONIC_DEFAULT_POS`（HA 序）—— 录制里逐位相等 ⇒ 直接比
    q_blk = out["obs994"][:, 94:384].reshape(len(idx), HIST, 29)
    out["q_block_matches_hist"] = float(np.abs(q_blk - out["q_hist"]).max())
    return out


# --------------------------------------------------------------------------- E1/E2
def exp_graph_and_token(dec, ort, frames, rows: list[dict], *, n_tok_dir: int = 8) -> dict:
    obs = frames["obs994"]
    a_t = dec(torch.from_numpy(obs)).detach().numpy()
    a_o = ort(obs)
    res = {
        "e1_max_abs": float(np.abs(a_t - a_o).max()),
        "e1_mean_abs": float(np.abs(a_t - a_o).mean()),
        "e1_vs_recorded_max_abs": float(np.abs(a_t - frames["raw_action"]).max()),
        "e1_vs_recorded_mean_abs": float(np.abs(a_t - frames["raw_action"]).mean()),
    }
    rows.append(dict(experiment="E1_graph_parity", variant="torch_vs_onnx",
                     metric="max_abs_action_diff", value=res["e1_max_abs"]))
    rows.append(dict(experiment="E1_graph_parity", variant="torch_vs_onnx",
                     metric="mean_abs_action_diff", value=res["e1_mean_abs"]))
    rows.append(dict(experiment="E1_graph_parity", variant="torch_vs_recorded_action",
                     metric="max_abs_action_diff", value=res["e1_vs_recorded_max_abs"]))
    rows.append(dict(experiment="E1_graph_parity", variant="torch_vs_recorded_action",
                     metric="mean_abs_action_diff", value=res["e1_vs_recorded_mean_abs"]))

    # E2：token 方向扰动
    rng = np.random.default_rng(0)
    cos_all, ratio_all = [], []
    for mag in (0.01, 0.03, 0.08):
        for k in range(n_tok_dir):
            delta = rng.normal(size=64).astype(np.float32)
            delta /= np.linalg.norm(delta)
            delta *= mag
            o1 = obs.copy(); o1[:, :64] += delta
            ap = dec(torch.from_numpy(o1)).detach().numpy() - a_t
            as_ = ort(o1) - a_o
            for i in range(len(obs)):
                cos_all.append(_cos(ap[i], as_[i]))
                ratio_all.append(float(np.linalg.norm(as_[i]) / max(np.linalg.norm(ap[i]), 1e-12)))
        rows.append(dict(experiment="E2_token_fd", variant=f"delta={mag:g}",
                         metric="cos_proxy_vs_sonic_mean", value=float(np.mean(cos_all[-len(obs) * n_tok_dir:]))))
        rows.append(dict(experiment="E2_token_fd", variant=f"delta={mag:g}",
                         metric="norm_ratio_sonic_over_proxy_mean",
                         value=float(np.mean(ratio_all[-len(obs) * n_tok_dir:]))))
    res["e2_cos_min"] = float(np.min(cos_all))
    res["e2_norm_ratio_range"] = [float(np.min(ratio_all)), float(np.max(ratio_all))]
    return res


# --------------------------------------------------------------------------- E3
def exp_obs_conventions(dec, ort, frames, rows: list[dict]) -> dict:
    """同一物理状态、不同 994 拼装口径：绝对动作误差 + Jacobian 方向。"""
    obs0 = frames["obs994"].copy()
    n = min(8, len(obs0))
    obs0 = obs0[:n]
    tok0 = obs0[:, :64].copy()
    a0 = ort(obs0)
    J0 = np.stack([_jac_token(ort, obs0[i], tok0[i]) for i in range(n)])

    variants: dict[str, np.ndarray] = {}

    # (i) 关节序错：把 q / qd / a_prev 三块按 ha→std 直接置换（模拟"以为 obs 是 standard 序"）
    o = obs0.copy()
    for s, e in (BLOCK_SLICES["q"], BLOCK_SLICES["qd"], BLOCK_SLICES["a_prev"]):
        blk = o[:, s:e].reshape(n, HIST, 29)
        o[:, s:e] = blk[:, :, P_HA2STD].reshape(n, HIST * 29)
    variants["joint_order_permuted"] = o

    # (ii) last_action 错一帧（把块整体后移一帧，最新一帧丢失）——文档记录的 top 风险
    o = obs0.copy()
    blk = o[:, 674:964].reshape(n, HIST, 29)
    sh = np.empty_like(blk)
    sh[:, 1:] = blk[:, :-1]
    sh[:, 0] = blk[:, 0]
    o[:, 674:964] = sh.reshape(n, HIST * 29)
    variants["last_action_shift1"] = o

    # (iii) q 偏移口径错：把 SONIC_DEFAULT 加回去（即按"decoder 要 raw q"的错误说法拼）
    o = obs0.copy()
    blk = o[:, 94:384].reshape(n, HIST, 29)
    o[:, 94:384] = (blk + SONIC_DEFAULT_POS_HA.reshape(1, 1, 29)).reshape(n, HIST * 29)
    variants["q_offset_raw"] = o

    # (iv) token 落到 FSQ 格点（部署 encoder 输出一定在格点上，训练 token_quant=none 不在）
    o = obs0.copy()
    o[:, :64] = (np.round(o[:, :64] * 16.0) / 16.0).astype(np.float32)
    variants["token_fsq_grid"] = o

    res = {}
    for name, ov in variants.items():
        av = ort(ov)
        Jv = np.stack([_jac_token(ort, ov[i], ov[i, :64]) for i in range(n)])
        cos_j = [_cos(J0[i], Jv[i]) for i in range(n)]
        d = av - a0
        res[name] = {
            "abs_action_err_mean": float(np.abs(d).mean()),
            "abs_action_err_max": float(np.abs(d).max()),
            "rms_action_err": float(np.sqrt((d ** 2).mean())),
            "jacobian_cos_mean": float(np.mean(cos_j)),
            "jacobian_cos_min": float(np.min(cos_j)),
            "jacobian_norm_ratio_mean": float(np.mean(
                [np.linalg.norm(Jv[i]) / max(np.linalg.norm(J0[i]), 1e-12) for i in range(n)])),
        }
        for m, v in res[name].items():
            rows.append(dict(experiment="E3_obs_convention", variant=name, metric=m, value=v))
    return res


# --------------------------------------------------------------------------- E4
def exp_delta_landing(dec, ort, frames, arms: list[tuple[str, Path]], rows: list[dict],
                      delta_root: Path) -> dict:
    """真实 Δtoken（正确 / 错误特征序）经过真实 decoder 后的动作修正。"""
    obs0 = frames["obs994"]
    state64 = frames["state64"]
    n = min(16, len(obs0))
    obs0, state64 = obs0[:n], state64[:n]
    tok0 = obs0[:, :64]
    a0 = ort(obs0)
    out = {}
    for name, path in arms:
        if not path.exists():
            out[name] = {"status": "MISSING"}
            continue
        d = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(d, dict) or d.get("delta_token_state") is None:
            out[name] = {"status": "NO_DELTA"}
            continue
        net, spec = DIO.load_delta_token_from_checkpoint(d)
        # 正确特征（含关节序置换）
        feats = np.stack([DIO.delta_features_from_state64(s, spec) for s in state64])
        deltas = np.stack([DIO.delta_from_token_chunk(tok0[i], feats[i], net, spec)
                           for i in range(n)])
        # 错误特征（不置换 = 旧部署口径）
        feats_w = (state64[:, :64] if spec.feat_dim == 64 else
                   np.concatenate([state64[:, :6], state64[:, 35:64]], 1))
        deltas_w = np.stack([DIO.delta_from_token_chunk(tok0[i], feats_w[i], net, spec)
                             for i in range(n)])

        def _landing(dl: np.ndarray) -> np.ndarray:
            o = obs0.copy()
            o[:, :64] = (tok0 + dl).astype(np.float32)
            return ort(o) - a0

        L = _landing(deltas)
        Lw = _landing(deltas_w)
        r = {
            "status": "OK",
            "spec": spec.as_dict(),
            "delta_token_norm_mean": float(np.linalg.norm(deltas, axis=1).mean()),
            "delta_action_norm_mean": float(np.linalg.norm(L, axis=1).mean()),
            "delta_action_rms_per_joint": float(np.sqrt((L ** 2).mean())),
            "gain_action_over_delta": float(np.linalg.norm(L, axis=1).mean()
                                            / max(np.linalg.norm(deltas, axis=1).mean(), 1e-12)),
            "cos_delta_action_correct_vs_wrong": float(np.mean(
                [_cos(L[i], Lw[i]) for i in range(n)])),
            "norm_ratio_wrong_over_correct": float(np.mean(
                [np.linalg.norm(Lw[i]) / max(np.linalg.norm(L[i]), 1e-12) for i in range(n)])),
            "max_joint_action_err_wrong": float(np.abs(Lw - L).max()),
        }
        out[name] = r
        for m in ("delta_token_norm_mean", "delta_action_norm_mean",
                  "delta_action_rms_per_joint", "gain_action_over_delta",
                  "cos_delta_action_correct_vs_wrong", "norm_ratio_wrong_over_correct",
                  "max_joint_action_err_wrong"):
            rows.append(dict(experiment="E4_delta_landing", variant=name, metric=m, value=r[m]))
        # 打印每个 checkpoint 的 Δ 范数（部署侧的"落地量"口径）
        print(f"[E4] {name}: ‖Δtoken‖={r['delta_token_norm_mean']:.5f} "
              f"‖Δa‖={r['delta_action_norm_mean']:.5f} "
              f"RMS/关节={r['delta_action_rms_per_joint']:.5f} "
              f"cos(correct,wrong)={r['cos_delta_action_correct_vs_wrong']:+.4f}", flush=True)
    return out


# --------------------------------------------------------------------------- E5
def exp_proxy_vs_deploy(frames, rows: list[dict], rows_e5: dict) -> dict:
    """训 930 观测 → SonicExecutor（训练执行器） vs 真实 ONNX（部署）。"""
    n = min(8, len(frames["obs994"]))
    obs994_ref = frames["obs994"][:n]
    # ★ 五个块**直接从录制的 994 里切**（``decoder_last_action_hist`` 与 obs 块差一帧，
    #   实测 max|Δ|=3.30 —— 用错数组会让"重建 994"自检直接失败 18 个单位，见 docs）
    q = obs994_ref[:, 94:384].reshape(n, HIST, 29)    # **HA 序**、已 −SONIC_DEFAULT
    qd = obs994_ref[:, 384:674].reshape(n, HIST, 29)
    la = obs994_ref[:, 674:964].reshape(n, HIST, 29)
    av = obs994_ref[:, 64:94].reshape(n, HIST, 3)
    g = obs994_ref[:, 964:994].reshape(n, HIST, 3)
    tok = obs994_ref[:, :64].copy()

    # HA → STANDARD（训练 obs 是 STANDARD 序）：`x_std = x_ha[p]`（p = ha→std 置换本身）
    inv = np.argsort(P_HA2STD)                       # std→ha 的索引
    q_std, qd_std, la_std = q[:, :, P_HA2STD], qd[:, :, P_HA2STD], la[:, :, P_HA2STD]
    # mjlab/SuperMimic 930：term-major，每项 10 帧连续
    obs930 = np.concatenate([q_std.reshape(n, -1), qd_std.reshape(n, -1),
                             av.reshape(n, -1), g.reshape(n, -1),
                             la_std.reshape(n, -1)], axis=1).astype(np.float32)
    ex = SonicExecutor(device="cpu")
    a_proxy_std = ex.act(torch.from_numpy(tok), torch.from_numpy(obs930)).detach().numpy()
    # 反置换回 HA 序，与 ONNX 直接比
    a_proxy_ha = a_proxy_std[:, inv]                 # `x_ha = x_std[argsort(p)]`
    a_onnx = OrtDecoder(DEFAULT_DECODER_ONNX)(obs994_ref)
    # 自检：代理拼出来的 994 是否与录制 994 一致
    obs_from_proxy = ex.obs994(torch.from_numpy(obs930), torch.from_numpy(tok)).numpy()
    res = {
        "obs994_rebuild_max_abs": float(np.abs(obs_from_proxy - obs994_ref).max()),
        "action_max_abs_diff_after_perm": float(np.abs(a_proxy_ha - a_onnx).max()),
        "action_max_abs_diff_no_perm": float(np.abs(a_proxy_std - a_onnx).max()),
        "order_trap_abs_scale": float(np.abs(a_proxy_std - a_onnx).max()
                                      / max(np.abs(a_proxy_ha - a_onnx).max(), 1e-12)),
    }
    for m, v in res.items():
        rows.append(dict(experiment="E5_proxy_vs_deploy", variant="sonic_executor_vs_onnx",
                         metric=m, value=v))
    rows_e5.update(res)
    return res


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/ha_raw/*/*/*.npz")
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--arms", nargs="+", default=[])
    ap.add_argument("--out", default="runs/harness_01_deploy/decoder_alignment.csv")
    ap.add_argument("--json", default="runs/harness_01_deploy/decoder_alignment.json")
    ap.add_argument("--root", default=str(DATA_ROOT))
    a = ap.parse_args()

    root = Path(a.root)
    files = sorted(glob.glob(str(root / a.npz), recursive=True))
    if not files:
        raise SystemExit(f"[align] 找不到录制：{root / a.npz}")
    npz = Path(files[0])
    frames = load_frames(npz, a.frames)
    print(f"[align] 录制={npz}\n[align] 帧={len(frames['idx'])} "
          f"q块自检 max|Δ|={frames['q_block_matches_hist']:.2e} "
          f"(obs[94:384] vs q_hist−SONIC_DEFAULT)", flush=True)
    print(f"[align] 录制里记录的 decoder 路径={frames['decoder_path_recorded']}", flush=True)
    print(f"[align] 实际加载的 ONNX={DEFAULT_DECODER_ONNX} "
          f"sha256={DIO.sha256_file(Path(DEFAULT_DECODER_ONNX))}", flush=True)

    dec = SonicDecoderTorch.from_onnx(str(DEFAULT_DECODER_ONNX)).eval()
    ort = OrtDecoder(DEFAULT_DECODER_ONNX)
    rows_e5: dict = {}
    rows: list[dict] = []
    res = {"npz": str(npz), "decoder_onnx": str(DEFAULT_DECODER_ONNX),
           "decoder_sha256": DIO.sha256_file(Path(DEFAULT_DECODER_ONNX)),
           "frames": int(len(frames["idx"])),
           "q_block_check_max_abs": frames["q_block_matches_hist"]}
    res.update(exp_graph_and_token(dec, ort, frames, rows))
    res["e3"] = exp_obs_conventions(dec, ort, frames, rows)
    arms = []
    for spec in a.arms:
        nm, _, rel = spec.partition("=")
        p = Path(rel)
        arms.append((nm, p if p.is_absolute() else root / p))
    if arms:
        res["e4"] = exp_delta_landing(dec, ort, frames, arms, rows, root)
    res["e5"] = exp_proxy_vs_deploy(frames, rows, rows_e5)

    print("\n[align] === E1 图级等价 ===")
    print(f"  torch vs ONNX  max|Δa|={res['e1_max_abs']:.3e}  mean|Δa|={res['e1_mean_abs']:.3e}")
    print(f"  torch vs 录制动作 max|Δa|={res['e1_vs_recorded_max_abs']:.3e} "
          f"mean={res['e1_vs_recorded_mean_abs']:.3e}")
    print(f"[align] === E2 token 扰动 ===  cos_min={res['e2_cos_min']:.6f} "
          f"norm_ratio={res['e2_norm_ratio_range']}")
    print("[align] === E3 obs 口径敏感度（绝对误差 / Jacobian 方向）===")
    for k, v in res["e3"].items():
        print(f"  {k:22s} |Δa|mean={v['abs_action_err_mean']:.4f} max={v['abs_action_err_max']:.4f} "
              f"J-cos={v['jacobian_cos_mean']:+.4f} (min {v['jacobian_cos_min']:+.4f}) "
              f"‖J‖比={v['jacobian_norm_ratio_mean']:.3f}")
    print("[align] === E5 训练执行器 vs 部署 decoder ===")
    for k, v in res["e5"].items():
        print(f"  {k:34s} = {v:.4e}")

    outp = Path(a.out); outp = outp if outp.is_absolute() else root / outp
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "variant", "metric", "value"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "value": f"{float(r['value']):.8g}"})
    jsonp = Path(a.json); jsonp = jsonp if jsonp.is_absolute() else root / jsonp
    jsonp.parent.mkdir(parents=True, exist_ok=True)
    jsonp.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[align] wrote {outp}\n[align] wrote {jsonp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
