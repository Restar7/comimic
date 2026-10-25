#!/usr/bin/env python3
"""HA（HumanoidArena）闭环推理的**锁死契约**：场景 / 资产 / 机器人初始位姿 / 仿真参数 / 编解码器。

以后任何人（或 agent）说"跑 HA 闭环推理"，都必须先过这里：

    python3 scripts/pin_ha_closedloop.py --check     # 校验，漂了就非零退出
    python3 scripts/pin_ha_closedloop.py --write     # 有意变更后重新固化（git diff 必须 review）

`--check` 校验四类东西，任一不符即失败：
  1. **文件级**：7 个任务的 yaml、env_cfg、被 import 的 scene 模块、场景/物体 USD、
     机器人 USD 的 sha256 + 大小；
  2. **数值级**：每个任务的 `ROBOT_INIT_POS/ROBOT_INIT_ROT`（或直接写的 init_pos/init_rot）、
     `sim.dt` / `decimation` / `wait_for_textures`；
  3. **资产存在性**：所有被引用的 `assets/**.usd*` 必须在盘上；
  4. **执行器/权重**：SONIC release encoder+decoder、默认 VLA / WBC ckpt 的 sha256。

改契约的唯一正确姿势：`--write` → `git diff configs/ha_closedloop_pin.json` → 确认每一项都是
有意的 → 提交。**不许**为了让 --check 过而手改 json。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/inspire/hdd/global_user/liumengfan-253108110079/wsc-workspace/comimic")
HA = ROOT / "third_party/HumanoidArena/isaaclab_twist2_g1"
PIN_PATH = Path(os.environ.get("HA_PIN_PATH", ROOT / "configs/ha_closedloop_pin.json"))

# key -> (task_id, yaml 相对 HA, env_cfg 相对 HA)
TASKS: dict[str, tuple[str, str, str]] = {
    "DoubleDesk": (
        "Isaac-Move-PickPlace-DoubleDesk-G129-Dex3-Wholebody",
        "tasks/common_test_config/base_test/doubledesk_sonic_test.yaml",
        "tasks/g1_tasks/move_pickplace_doubledesk_g1_29dof_dex3_wholebody/move_pickplace_doubledesk_g1_29dof_dex3_hw_env_cfg.py",
    ),
    "PPBox": (
        "Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby",
        "tasks/common_test_config/base_test/pp_box_sonic_test.yaml",
        "tasks/g1_tasks/move_pickplace_box_g1_29dof_dex3_wholedoby/move_pickplace_box_g1_29dof_dex3_hw_env_cfg.py",
    ),
    "Boxing": (
        "Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody",
        "tasks/common_test_config/base_test/boxing_sonic_test.yaml",
        "tasks/g1_tasks/move_boxing_bag_g1_29dof_dex3_wholebody/move_boxing_bag_g1_29dof_dex3_hw_env_cfg.py",
    ),
    "OpenDoor": (
        "Isaac-Move-Open-Door-G129-Dex3-Wholebody",
        "tasks/common_test_config/base_test/open_door_sonic_test.yaml",
        "tasks/g1_tasks/move_open_door_g1_29dof_dex3_wholebody/move_open_door_g1_29dof_dex3_hw_env_cfg.py",
    ),
    "SitSofa": (
        "Isaac-Move-Sit-Sofa-G129-Dex3-Wholebody",
        "tasks/common_test_config/base_test/sit_sofa_sonic_test.yaml",
        "tasks/g1_tasks/move_sit_sofa_g1_29dof_dex3_wholebody/move_sit_sofa_g1_29dof_dex3_hw_env_cfg.py",
    ),
    "VisionNavi": (
        "Isaac-Move-SmallWarehouse-VisionNavigation-G129-Dex3-Wholebody",
        "tasks/common_test_config/base_test/vision_navi_sonic_test.yaml",
        "tasks/g1_tasks/move_small_warehouse_vision_navigation_g1_29dof_dex3_wholebody/move_small_warehouse_vision_navigation_g1_29dof_dex3_hw_env_cfg.py",
    ),
    "FootballSingle": (
        "Isaac-Move-Football-Single-G129-Dex3-Wholebody",
        "tasks/common_test_config/base_test/football_single_sonic_test.yaml",
        "tasks/g1_tasks/move_football_single_g1_29dof_dex3_wholebody/move_football_single_g1_29dof_dex3_hw_env_cfg.py",
    ),
}

ROBOT_TYPE = "unitree_g1_refpose_v3_1"
ROBOT_USD_DIR = "assets/robots/g1-29dof-dex3-base-fix-usd"
SONIC_RELEASE = ROOT / "../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release"
DEFAULT_VLA = ROOT / "runs/groot_sonic_vla_ha_b64_lr1e-5/checkpoint-8000"
DEFAULT_WBC = ROOT / "trackerlab-supermimic/logs/bones_seed_token_ha_v2/ckpt_010500_physint.pt"
PROTOCOL = {"seeds": list(range(10)), "repeats_per_seed": 1, "max_steps": 300, "video_fps": 30,
            "headless": True, "episode_seed": "seed 直通（seed i -> episode_seed i）"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def file_entry(path: Path) -> dict:
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256(path)}


def strip_comments(text: str) -> str:
    return "\n".join(line.split("#")[0] for line in text.splitlines())


def usd_refs(path: Path) -> list[Path]:
    text = re.sub(r'"\s*\n\s*f?"', "", strip_comments(path.read_text(errors="ignore")))
    found = set(re.findall(r'assets/[^"\']+?\.usd[ac]?', text))
    return [HA / f for f in sorted(found)]


def scene_modules(env_cfg: Path) -> list[Path]:
    text = env_cfg.read_text(errors="ignore")
    names = set(re.findall(r"from\s+tasks\.common_scene\.(\w+)\s+import", text))
    return [HA / "tasks/common_scene" / f"{n}.py" for n in sorted(names)]


def first_match(patterns: list[str], text: str) -> str | None:
    for pat in patterns:
        m = re.search(pat, text, re.M)
        if m:
            return m.group(1).strip()
    return None


def resolve_symbols(value: str | None, texts: list[str]) -> str | None:
    """把 `ROBOT_INIT_X, ROBOT_INIT_Y-2, ROBOT_INIT_Z` 这种常量位姿解析成数字。

    常量可能写在 env_cfg 自己文件里，也可能写在它 import 的 scene 模块里（football 是后者），
    所以要把这些文件的文本一起传进来。只对纯数字表达式求值，出现别的标识符就原样返回。
    """
    if not value:
        return value
    consts: dict[str, float] = {}
    for t in texts:
        consts.update({n: float(v) for n, v in re.findall(r"^(ROBOT_INIT_[XYZ])\s*=\s*(-?[0-9.eE+]+)", t, re.M)})
    out = value
    for name, num in consts.items():
        out = re.sub(rf"\b{name}\b", repr(num), out)
    vals = []
    for part in out.strip().strip("()").split(","):
        if not re.fullmatch(r"[\s0-9eE+\-*/().]+", part):
            return out
        try:
            vals.append(round(float(eval(part.strip(), {"__builtins__": {}}, {})), 6))
        except Exception:  # noqa: BLE001
            return out
    return "(" + ", ".join(repr(v) for v in vals) + ")"


def build() -> dict:
    tasks: dict[str, dict] = {}
    for key, (task_id, yaml_rel, cfg_rel) in TASKS.items():
        yaml_path, cfg_path = HA / yaml_rel, HA / cfg_rel
        if not yaml_path.is_file():
            raise SystemExit(f"[pin] 缺 yaml: {yaml_path}")
        if not cfg_path.is_file():
            raise SystemExit(f"[pin] 缺 env_cfg: {cfg_path}")
        ytxt = strip_comments(yaml_path.read_text(errors="ignore"))
        ctxt = cfg_path.read_text(errors="ignore")
        mods = scene_modules(cfg_path)
        assets: dict[str, Path] = {}
        for m in mods:
            for f in usd_refs(m):
                assets[str(f)] = f
        for f in usd_refs(cfg_path):
            assets[str(f)] = f
        missing = [str(p) for p in assets.values() if not p.is_file()]
        tasks[key] = {
            "task_id": task_id,
            "yaml": file_entry(yaml_path),
            "env_cfg": file_entry(cfg_path),
            "scene_modules": [file_entry(m) for m in mods],
            "init_pos": resolve_symbols(
                first_match([r"ROBOT_INIT_POS\s*=\s*\(([^)]*)\)", r"init_pos\s*=\s*\(([^)]*)\)"], ctxt),
                [ctxt, *[m.read_text(errors="ignore") for m in mods]]),
            "init_rot": resolve_symbols(
                first_match([r"ROBOT_INIT_ROT\s*=\s*\(([^)]*)\)", r"init_rot\s*=\s*\(([^)]*)\)"], ctxt),
                [ctxt, *[m.read_text(errors="ignore") for m in mods]]),
            "sim": {
                "dt": first_match([r"^\s*dt:\s*([0-9.eE+-]+)"], ytxt),
                "decimation": first_match([r"^\s*decimation:\s*([0-9]+)"], ytxt),
                "wait_for_textures": first_match([r"wait_for_textures:\s*(\w+)"], ytxt),
            },
            "object_resets": re.findall(r"record_name:\s*(\S+)", ytxt),
            "assets": [file_entry(p) for p in sorted(assets.values())],
            "missing_assets": missing,
        }

    robot_files = sorted(p for p in (HA / ROBOT_USD_DIR).rglob("*") if p.is_file() and p.suffix in (".usd", ".usda", ".usdc"))
    return {
        "version": 1,
        "purpose": "HA 闭环推理锁死契约：场景/资产/机器人初始位姿/仿真参数/编解码器",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ha_root": str(HA),
        "robot_type": ROBOT_TYPE,
        "protocol": PROTOCOL,
        "runtime": {
            "script": str((HA / "script/eval_scripts/sonic/sim_eval_vla.py").resolve()),
            "script_sha256": sha256(HA / "script/eval_scripts/sonic/sim_eval_vla.py"),
            "episode_batch_mode": "--episode_batch_json（单进程持久仿真，多集复用同一场景）",
        },
        "weights": {
            "sonic_encoder": file_entry(SONIC_RELEASE / "model_encoder.onnx"),
            "sonic_decoder": file_entry(SONIC_RELEASE / "model_decoder.onnx"),
            "vla_default": {"path": str(DEFAULT_VLA.resolve())},
            "wbc_default": file_entry(DEFAULT_WBC),
        },
        "robot_usd": [file_entry(p) for p in robot_files],
        "robot": {
            "type": ROBOT_TYPE,
            "usd_dir": str((HA / ROBOT_USD_DIR).resolve()),
            "cfg": file_entry(HA / "tasks/common_config/robot_configs.py"),
            "default_joint_source": "tasks/common_config/robot_configs.py::G1RobotJointTemplates（随 cfg 的 sha256 锁定）",
            "pose_overrides_env": "ROBOT_USD_OVERRIDE 一旦设置就会换机器人 USD —— 契约里禁止使用",
        },
        "tasks": tasks,
    }


def check(pin: dict) -> int:
    cur = build()
    problems: list[str] = []

    def cmp_file(label: str, rec: dict, now: dict | None):
        if now is None:
            problems.append(f"[缺文件] {label}: {rec['path']}")
        elif rec.get("sha256") != now.get("sha256"):
            problems.append(f"[内容变了] {label}: {rec['path']}\n            {rec['sha256'][:16]} -> {now['sha256'][:16]}")

    # 全局文件
    for sect in ("weights",):
        for name, rec in pin.get(sect, {}).items():
            if isinstance(rec, dict) and "sha256" in rec:
                cmp_file(f"weights.{name}", rec, cur["weights"].get(name))
    if "cfg" in (pin.get("robot") or {}):
        cmp_file("robot.cfg", pin["robot"]["cfg"], cur["robot"]["cfg"])
    if (pin.get("robot") or {}).get("type") != cur["robot"]["type"]:
        problems.append(f"[机器人换了] {pin['robot'].get('type')} -> {cur['robot']['type']}")
    import os as _os
    if _os.environ.get("ROBOT_USD_OVERRIDE"):
        problems.append(f"[环境变量违规] ROBOT_USD_OVERRIDE={_os.environ['ROBOT_USD_OVERRIDE']} 会替换机器人资产")
    for rec in pin.get("robot_usd", []):
        now = next((r for r in cur["robot_usd"] if r["path"] == rec["path"]), None)
        cmp_file("robot_usd", rec, now)

    # 每个任务
    for key, old in pin.get("tasks", {}).items():
        new = cur["tasks"].get(key)
        if new is None:
            problems.append(f"[缺任务] {key}")
            continue
        for field in ("yaml", "env_cfg"):
            cmp_file(f"{key}.{field}", old[field], new[field])
        old_mods = {m["path"]: m for m in old.get("scene_modules", [])}
        new_mods = {m["path"]: m for m in new.get("scene_modules", [])}
        for p, rec in old_mods.items():
            cmp_file(f"{key}.scene_module", rec, new_mods.get(p))
        if set(old_mods) != set(new_mods):
            problems.append(f"[scene 模块集合变了] {key}: {sorted(set(old_mods) ^ set(new_mods))}")
        old_assets = {a["path"]: a for a in old.get("assets", [])}
        new_assets = {a["path"]: a for a in new.get("assets", [])}
        for p, rec in old_assets.items():
            cmp_file(f"{key}.asset", rec, new_assets.get(p))
        for p in set(new_assets) - set(old_assets):
            problems.append(f"[新增资产未固化] {key}: {p}")
        for field in ("init_pos", "init_rot"):
            if old.get(field) != new.get(field):
                problems.append(f"[初始位姿变了] {key}.{field}: {old.get(field)!r} -> {new.get(field)!r}")
        for k, v in (old.get("sim") or {}).items():
            if (new.get("sim") or {}).get(k) != v:
                problems.append(f"[仿真参数变了] {key}.sim.{k}: {v!r} -> {(new.get('sim') or {}).get(k)!r}")
        if (old.get("object_resets") or []) != (new.get("object_resets") or []):
            problems.append(f"[物体 reset 项变了] {key}: {old.get('object_resets')} -> {new.get('object_resets')}")
        if new.get("missing_assets"):
            problems.append(f"[资产缺失] {key}: {new['missing_assets']}")
    if (pin.get("runtime") or {}).get("script_sha256") != cur["runtime"]["script_sha256"]:
        problems.append("[评测脚本变了] script/eval_scripts/sonic/sim_eval_vla.py —— 结论口径可能变，需重新固化")

    if problems:
        print(f"[pin] ❌ 契约校验失败（{len(problems)} 处）：", file=sys.stderr)
        for p in problems:
            print("   " + p, file=sys.stderr)
        print("\n[pin] 如果这些变更是**有意**的：python3 scripts/pin_ha_closedloop.py --write "
              "然后 review `git diff configs/ha_closedloop_pin.json`。", file=sys.stderr)
        return 1
    print(f"[pin] ✅ 契约校验通过：{len(pin['tasks'])} 任务 / "
          f"{sum(len(t['assets']) for t in pin['tasks'].values())} 个资产 / "
          f"init pose 已锁 / decoder=release")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="重新生成锁死契约")
    ap.add_argument("--check", action="store_true", help="校验是否漂移（漂了非零退出）")
    ap.add_argument("--quiet", action="store_true", help="通过时不打印")
    args = ap.parse_args()

    if args.write:
        PIN_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = build()
        PIN_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print(f"[pin] 已固化 -> {PIN_PATH}")
        print(f"[pin] 任务 {len(data['tasks'])} 个；资产 "
              f"{sum(len(t['assets']) for t in data['tasks'].values())} 个；"
              f"机器人 USD {len(data['robot_usd'])} 个")
        return

    if not PIN_PATH.is_file():
        raise SystemExit(f"[pin] 契约文件不存在：{PIN_PATH}\n先跑 --write")
    pin = json.loads(PIN_PATH.read_text())
    rc = check(pin)
    if rc == 0 and not args.quiet:
        print(f"[pin] 契约文件：{PIN_PATH}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
