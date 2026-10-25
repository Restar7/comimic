#!/usr/bin/env python3
"""B3: HTTP inference server that exposes **our GR00T VLA** to HumanoidArena's evaluator.

HumanoidArena's eval is a client-server design: the Isaac Lab client posts an observation to
``POST /infer`` and expects a chunk of motions back (``latent64_chunk``).  The official
implementation of that server is **not** released, so this file is ours.

Wire format (verbatim from ``action_provider/lerobot_vla_http_client.py``)::

    POST /infer
    {
      "observation": {
        "images": {"front": {"shape": [H, W, 3], "dtype": "uint8", "data_b64": "<base64 raw>"}},
        "state": [64 floats]                          # vla_state (rot6d+joint+vel, rotlocal v3)
      },
      "robot_type": "unitree_g1_refpose_v3_1",
      "return_chunk": true,
      "task": "<instruction>"
    }
    -> {"latent64_chunk": [[64 floats] x N], "meta": {...}}      # N = VLA token horizon (40)

    POST /reset   {"seed": 0} -> {"ok": true}

Why ``latent64_chunk``: our VLA emits a ``(40, 132)`` flow-matching chunk; columns ``0:64`` are
the SONIC motion token, which is exactly the ``latent64`` the SONIC/HA stack consumes and what
our SuperMimic action provider (``action_provider_supermimic.py``) needs.

Two run modes
-------------
``--dry-run``  no model: returns a deterministic synthetic chunk. Used to validate the client
               protocol end-to-end without a GPU (and in CI).
default        loads the real GR00T VLA from ``--ckpt`` (needs the GR00T processor + weights).

Examples
--------
    # protocol self-test (server + client, no GPU)
    python3 scripts/ha_vla_server.py --self-test

    # real server for the A1 arms
    python3 scripts/ha_vla_server.py --ckpt runs/ckpt_vla_ft4000 --port 18080 --device cuda:0
    python3 scripts/ha_vla_server.py --ckpt runs/ckpt_vla20k   --port 18081 --device cuda:1
"""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- policy backends
class DummyPolicy:
    """Deterministic synthetic VLA: enough to exercise the wire protocol."""

    def __init__(self, horizon: int = 40, token_dim: int = 64):
        self.horizon, self.token_dim = horizon, token_dim

    def infer(self, front_rgb: np.ndarray, state: np.ndarray, task: str | None) -> np.ndarray:
        seed = float(np.asarray(state, dtype=np.float64).sum()) + float(front_rgb.mean())
        rng = np.random.default_rng(int(abs(seed)) % (2**32))
        return rng.normal(0, 0.1, (self.horizon, self.token_dim)).astype(np.float32)


class Gr00tVlaPolicy:
    """Real GR00T VLA: RGB + vla_state + instruction → ``(horizon, token_dim)`` motion tokens."""

    def __init__(self, ckpt: str, vlm_path: str | None, device: str, horizon: int = 40,
                 token_dim: int = 64, num_steps: int = 2):
        import torch  # noqa: PLC0415

        from diff_vla_wbc.gr00t_vlm import Gr00tVLM  # noqa: PLC0415

        self.torch = torch
        self.device = torch.device(device)
        self.horizon, self.token_dim = horizon, token_dim
        self.vlm = Gr00tVLM.from_pretrained(
            vlm_path, checkpoint=ckpt, device=device, action_horizon=horizon,
            freeze_vlm=True, attn_implementation="sdpa",
        )
        self.vlm.eval()
        self.num_steps = num_steps
        self.processor = None
        try:  # reuse the checkpoint's own preprocessor (statistics/transform come from it)
            from diff_vla_wbc.g1_tasks import load_gr00t_processor  # noqa: PLC0415

            self.processor = load_gr00t_processor(ckpt, vlm_path=vlm_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[server] warning: no GR00T processor ({exc}); using pass-through inputs",
                  flush=True)

    def infer(self, front_rgb: np.ndarray, state: np.ndarray, task: str | None) -> np.ndarray:
        import torch  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        from diff_vla_wbc.gr00t_fm import sample_model_with_grad  # noqa: PLC0415

        image = Image.fromarray(np.asarray(front_rgb, dtype=np.uint8))
        instruction = task or "Pick up the object and place it at the target."
        if self.processor is not None:
            inputs = self.processor(
                images=[image], language=[instruction],
                states=torch.as_tensor(np.asarray(state, dtype=np.float32)).reshape(1, -1),
            )
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        else:
            raise RuntimeError("server was built without a processor; cannot run the real VLA")
        noise = torch.randn(1, self.horizon, int(self.vlm.action_head.action_dim),
                            device=self.device, dtype=self.vlm.dtype)
        with torch.enable_grad():
            tokens = sample_model_with_grad(self.vlm, inputs, initial_noise=noise,
                                            num_steps=self.num_steps)
        return tokens.detach()[0, :, : self.token_dim].float().cpu().numpy()


# --------------------------------------------------------------------------- HTTP server
class _Handler(BaseHTTPRequestHandler):
    policy = None
    stats = {"infer": 0, "reset": 0, "errors": 0}

    def log_message(self, fmt, *args):  # keep the console usable
        if os.environ.get("HA_VLA_SERVER_VERBOSE", "0") == "1":
            super().log_message(fmt, *args)

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode() or "{}")
        except json.JSONDecodeError as exc:
            self._send(400, {"error": f"bad json: {exc}"})
            return

        path = self.path.rstrip("/")
        if path.endswith("/reset"):
            self.stats["reset"] += 1
            self._send(200, {"ok": True, "seed": payload.get("seed")})
            return
        if not path.endswith("/infer"):
            self._send(404, {"error": f"unknown path {self.path}"})
            return

        try:
            obs = payload.get("observation", {})
            img = obs.get("images", {}).get("front", {})
            if "data_b64" not in img:
                raise ValueError("observation.images.front.data_b64 is required")
            shape = tuple(int(x) for x in img.get("shape", []))
            buf = base64.b64decode(img["data_b64"])
            rgb = np.frombuffer(buf, dtype=np.uint8).reshape(shape) if shape else \
                np.frombuffer(buf, dtype=np.uint8)
            state = np.asarray(obs.get("state", []), dtype=np.float32).reshape(-1)
            tokens = self.policy.infer(rgb, state, payload.get("task"))
            self.stats["infer"] += 1
            self._send(200, {
                "latent64_chunk": np.asarray(tokens, dtype=np.float32).tolist(),
                "meta": {
                    "robot_type": payload.get("robot_type"),
                    "chunk": int(np.asarray(tokens).shape[0]),
                    "dim": int(np.asarray(tokens).shape[1]),
                    "infer_calls": self.stats["infer"],
                },
            })
        except Exception as exc:  # noqa: BLE001
            self.stats["errors"] += 1
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("/health"):
            self._send(200, {"ok": True, **self.stats})
        else:
            self._send(404, {"error": "use POST /infer, POST /reset, GET /health"})


def serve(policy, host: str, port: int) -> ThreadingHTTPServer:
    _Handler.policy = policy
    httpd = ThreadingHTTPServer((host, port), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


# --------------------------------------------------------------------------- self-test
def _self_test() -> None:
    """Spin a dry-run server and talk to it via the *real* HA client (protocol proof)."""
    httpd = serve(DummyPolicy(), "127.0.0.1", 0)
    port = httpd.server_address[1]
    try:
        client_path = (ROOT / "third_party" / "HumanoidArena" / "isaaclab_twist2_g1")
        sys.path.insert(0, str(client_path))
        from action_provider.lerobot_vla_http_client import LeRobotVLAHttpClient  # noqa: PLC0415

        client = LeRobotVLAHttpClient(f"http://127.0.0.1:{port}", timeout_s=5.0)
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        state = np.linspace(0, 1, 64, dtype=np.float32)
        chunk = client.infer_chunk(rgb, state, "unitree_g1_refpose_v3_1", task="pick the box")
        assert chunk.shape == (40, 64), chunk.shape
        client.reset(seed=0)
        chunk2 = client.infer_chunk(rgb, state, "unitree_g1_refpose_v3_1")
        assert chunk2.shape == (40, 64)
        assert np.allclose(chunk, chunk2), "dry-run policy must be deterministic"
        print(f"[ok] /infer -> latent64 chunk {chunk.shape}; /reset ok; deterministic")
    finally:
        httpd.shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, help="GR00T checkpoint dir (or raw run) to serve")
    ap.add_argument("--vlm-path", type=Path, default=ROOT / "groot_repro/nvidia/Cosmos-Reason2-2B-weights")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--horizon", type=int, default=40, help="VLA token horizon (HA expects 40)")
    ap.add_argument("--token-dim", type=int, default=64)
    ap.add_argument("--num-steps", type=int, default=2, help="FM Euler steps")
    ap.add_argument("--dry-run", action="store_true", help="no model, synthetic chunk (protocol only)")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return

    if args.dry_run:
        policy = DummyPolicy(args.horizon, args.token_dim)
        print("[server] dry-run policy (no model)", flush=True)
    else:
        if args.ckpt is None:
            raise SystemExit("--ckpt is required (or use --dry-run / --self-test)")
        policy = Gr00tVlaPolicy(str(args.ckpt), str(args.vlm_path) if args.vlm_path else None,
                                args.device, horizon=args.horizon, token_dim=args.token_dim,
                                num_steps=args.num_steps)
        print(f"[server] GR00T VLA loaded from {args.ckpt} on {args.device}", flush=True)

    httpd = serve(policy, args.host, args.port)
    print(f"[server] listening on http://{args.host}:{args.port}  "
          f"(POST /infer, POST /reset, GET /health)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
