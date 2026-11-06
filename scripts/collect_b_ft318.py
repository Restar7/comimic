#!/usr/bin/env python3
"""B-route collection: VLA in the loop → SONIC decoder → MuJoCo (on-policy episodes).

`docs/MUJOCO_SONIC_JOINTFT_PLAN_20261104.md` §5 的第 2 步。两种 token 源：

``--token-source dry``
    直接用 318 数据集里该帧的 ``action.motion_token``（**冒烟/对照**：等价于 A 线的长时回放，
    已知会漂/摔，见计划 §2.2；只用来验证采集器本身跑得通）。

``--token-source vla``
    POST 到 VLA 服务（协议同 `scripts/ha_vla_latent64_server.py` 头注释）::

        POST /infer  {"observation": {"images": {"front": {...}}, "state": [64 floats]},
                      "robot_type": "unitree_g1_refpose_v3_1", "return_chunk": true,
                      "task": "<instruction>"}
        -> {"latent64_chunk": [[64 floats], ...]}

    图像取数据集 ``videos/chunk-*/observation.images.ego_view/episode_XXXXXX.mp4`` 的同一帧
    （318 的录制视角就是训练分布，不需要在 MuJoCo 里重建场景）。

产物与 `scripts/build_real_transitions_mjlab.py` 的输入约定一致::

    states (N,71) / next_states (N,71) / actions (N,29) / frames (N,)
    + meta（episode、长度、追踪误差、是否摔）

用法::

    PY=third_party/SuperMimic-Reproduced/.venv/bin/python
    # 冒烟（不连 VLA）：每集 60 步
    $PY scripts/collect_b_ft318.py --set-index 0 --episodes 0 --max-steps 60 \
        --token-source dry --out runs/B_ft318_smoke.pt
    # 正式（VLA 在环）
    $PY scripts/collect_b_ft318.py --set-index 0 --episodes 0,1 --token-source vla \
        --vla-url http://127.0.0.1:18100 --out runs/B_ft318_r1/set0.pt
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "groot_repro"))


def _replay_module():
    spec = importlib.util.spec_from_file_location(
        "replay_ft318_mujoco", ROOT / "scripts" / "replay_ft318_mujoco.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def state71(plant) -> np.ndarray:
    d = plant.data
    return np.concatenate([
        d.qpos[0:3], d.qpos[3:7], plant.q(), d.qvel[0:3], d.qvel[3:6], plant.dq(),
    ]).astype(np.float32)


def state64(plant) -> list[float]:
    """HA ``vla_state`` = root rot6d(6) + dof_pos(29) + dof_vel(29)."""
    import g1_sim as G

    rot = G.quat_to_rotmat(plant.quat())
    rot6d = rot[:, :2].reshape(-1).tolist()      # 与 HA 的 quat_to_rot6d_wxyz 同序
    return rot6d + plant.q().tolist() + plant.dq().tolist()


class VideoFrames:
    """Lazy ego-view frames of one episode (dataset videos are the VLA's input distribution)."""

    def __init__(self, path: Path | None):
        self.path = path
        self._cap = None
        self._frames: dict[int, np.ndarray] = {}
        if path is None:
            return
        self._cv2 = self._import_cv2()
        if self._cv2 is None:
            print(f"[warn] cv2 不可用，图像用零帧代替（VLA 臂请先装 opencv）：{path}")

    @staticmethod
    def _import_cv2():
        """mjlab venv 里没有 cv2，但系统 python3.12 的 dist-packages 有（同 3.12 ABI）。"""
        try:
            import cv2  # noqa: PLC0415

            return cv2
        except Exception:
            pass
        import sys as _sys

        for extra in ("/usr/local/lib/python3.12/dist-packages", "/usr/lib/python3/dist-packages"):
            if extra not in _sys.path:
                _sys.path.append(extra)
            try:
                import cv2  # noqa: PLC0415

                return cv2
            except Exception:
                continue
        return None

    def get(self, idx: int) -> np.ndarray | None:
        if self.path is None or self._cv2 is None:
            return None
        if idx in self._frames:
            return self._frames[idx]
        if self._cap is None:
            self._cap = self._cv2.VideoCapture(str(self.path))
        cap = self._cap
        cap.set(self._cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            return None
        self._frames[idx] = frame
        return frame


def vla_infer(url: str, task: str, image: np.ndarray | None, state: list[float],
              timeout: float = 30.0) -> np.ndarray:
    import requests  # noqa: PLC0415

    obs = {"state": state}
    if image is not None:
        img = np.ascontiguousarray(image[:, :, ::-1] if image.shape[-1] == 3 else image)
        obs["images"] = {"front": {"shape": list(img.shape), "dtype": "uint8",
                                   "data_b64": base64.b64encode(img.tobytes()).decode()}}
    payload = {"observation": obs, "robot_type": "unitree_g1_refpose_v3_1",
               "return_chunk": True, "task": task}
    r = requests.post(url.rstrip("/") + "/infer", json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    for key in ("latent64_chunk", "action_chunk", "encoder_latent_chunk", "latent64", "encoder_latent"):
        if key in body:
            return np.asarray(body[key], dtype=np.float64)
    raise RuntimeError(f"VLA 响应里没有 token 字段，拿到：{sorted(body)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set-index", type=int, default=0)
    ap.add_argument("--episodes", default="0")
    ap.add_argument("--max-steps", type=int, default=0, help="0 = 跑到该集末尾")
    ap.add_argument("--token-source", default="dry", choices=("dry", "vla"))
    ap.add_argument("--vla-url", default="http://127.0.0.1:18100")
    ap.add_argument("--gains", default="ha", choices=("ha", "deploy"))
    ap.add_argument("--init", default="recorded", choices=("rest", "recorded"))
    ap.add_argument("--decoder", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/B_ft318_smoke.pt")
    args = ap.parse_args()

    import g1_sim as G
    from sonic_loop import Decoder, SIGNALS

    rp = _replay_module()
    plant = rp.Plant(rp.DEFAULT_SCENE, gains=args.gains)
    decoder = Decoder(str(args.decoder or rp.DEFAULT_DECODER))
    ha_to_std = np.array(
        __import__("joint_contract").permutation(src="ha_v31", dst="standard"), dtype=np.int64
    )
    scale = G.ACTION_SCALE.astype(np.float64)

    # 数据集目录（含 task instruction / videos）
    import json as _json

    manifest = _json.loads((ROOT / "data" / "sonic_datasets.json").read_text())
    active = manifest["sets"][manifest["active"]]
    root = Path(active["root"])
    sets = [(d["task_id"], d["instruction"], root / d["path"]) for d in active["datasets"]]
    task_id, instruction, ds_dir = sets[args.set_index]

    S, N, A, F, EP = [], [], [], [], []
    summary = []
    for epi in [int(x) for x in args.episodes.split(",") if x.strip()]:
        ep = rp.load_episode(args.set_index, epi)
        T = ep["state"].shape[0]
        n = T - 1 if args.max_steps <= 0 else min(args.max_steps, T - 1)
        video = ds_dir / "videos" / "chunk-000" / "observation.images.ego_view" / f"episode_{epi:06d}.mp4"
        frames = VideoFrames(video if video.is_file() else None)

        init = rp.frame_init_state(plant, ep, 0) if args.init == "recorded" else {}
        plant.reset(ep["body_state"][0], ep["hands_state"][0],
                    quat=init.get("quat", (1, 0, 0, 0)), dq=init.get("dq"),
                    base_lin_vel=init.get("base_lin_vel"), base_ang_vel=init.get("base_ang_vel"))
        hist = [{
            "base_ang_vel": np.zeros(3),
            "q": (ep["body_state"][0] - G.DEFAULT_ANGLES)[G.MUJOCO_TO_ISAACLAB],
            "dq": np.zeros(29),
            "last_action": np.zeros(29),
            "gravity": G.projected_gravity(plant.quat()),
        } for _ in range(10)]

        chunk: np.ndarray | None = None
        step_in_chunk = 0
        fell = False
        mae = []
        for t in range(n):
            if args.token_source == "dry":
                token = np.asarray(ep["tokens"][t], np.float64)
            else:
                if chunk is None or step_in_chunk >= len(chunk):
                    img = frames.get(t)
                    chunk = vla_infer(args.vla_url, instruction, img, state64(plant))
                    step_in_chunk = 0
                token = np.asarray(chunk[step_in_chunk], np.float64)
                step_in_chunk += 1

            parts = [token]
            for name in SIGNALS:
                parts.append(np.concatenate([f[name] for f in hist]))
            a_ha = decoder(np.concatenate(parts))
            a_std = a_ha[ha_to_std]
            s_before = state71(plant)
            plant.apply_target(G.DEFAULT_ANGLES + scale * a_std, ep["hands_wbc"][t])
            s_after = state71(plant)

            hist.append({
                "base_ang_vel": plant.data.qvel[3:6].copy(),
                "q": (plant.q() - G.DEFAULT_ANGLES)[G.MUJOCO_TO_ISAACLAB],
                "dq": plant.dq()[G.MUJOCO_TO_ISAACLAB],
                "last_action": a_ha,
                "gravity": G.projected_gravity(plant.quat()),
            })
            hist = hist[-10:]
            S.append(s_before)
            N.append(s_after)
            A.append(a_std.astype(np.float32))
            F.append(t)
            EP.append(int(epi))
            mae.append(float(np.abs(plant.q() - ep["body_state"][t + 1]).mean()))
            if plant.up() < 0.5:
                fell = True
                break

        row = {
            "episode": int(epi), "steps": len(mae), "planned": int(n), "fell": fell,
            "joint_mae": float(np.mean(mae)) if mae else None,
            "tilt_end_deg": plant.tilt_deg(),
            "root_xy": np.round(plant.root()[:2], 3).tolist(),
        }
        summary.append(row)
        print(f"[B] {task_id}/ep{epi}: {json.dumps(row, ensure_ascii=False)}", flush=True)

    if not S:
        raise SystemExit("[fail] 没有采到任何转移")
    payload = {
        "states": torch.tensor(np.stack(S)), "next_states": torch.tensor(np.stack(N)),
        "actions": torch.tensor(np.stack(A)), "frames": torch.tensor(F, dtype=torch.long),
        "episodes": torch.tensor(EP, dtype=torch.long),
        "meta": {"source": "collect_b_ft318", "token_source": args.token_source,
                 "set_index": args.set_index, "task": task_id,
                 "instruction": instruction, "gains": args.gains, "init": args.init,
                 "n": len(S), "episodes": summary},
    }
    if not (torch.isfinite(payload["states"]).all() and torch.isfinite(payload["actions"]).all()):
        raise SystemExit("[fail] NaN/Inf in collected transitions")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(args.out))
    print(f"[saved] {args.out}  n={len(S)}  frames={payload['frames'].min().item()}..{payload['frames'].max().item()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
