"""奖励口径的**唯一实现**（2026-11-05）。

为什么要单独一个模块
--------------------
奖励口径一共有 12 个字段（`reward_mode / vel_scale / huber_delta / huber_cap /
huber_err_d_max / delta_pos / delta_vel / delta_ori / root_ori_mode / tail_weight /
tail_cap_slope / yaw_hinge_lambda`）。如果「训练」和「补算奖励」各写一份，
迟早会分叉 —— 而分叉的表现是"critic 训得很好，但和策略优化的不是同一个目标"，
这种错**不会报错**，只会让结果莫名其妙。

所以：训练侧 `train_joint_v2.py` 与补算侧 `compute_collect_reward.py` **都调这一个函数**，
并且 `compute_collect_reward.py --from-args` 直接把某一轮的 `args.json` 灌进来，
保证"补算的奖励 == 那一轮训练时真正用的奖励"。
"""
from __future__ import annotations


def _getter(cfg):
    if isinstance(cfg, dict):
        return lambda k, d=None: cfg.get(k, d)
    return lambda k, d=None: getattr(cfg, k, d)


def apply_reward_cfg(bundle, cfg) -> None:
    """把奖励口径写到 ``bundle`` 上。``cfg`` 可以是 argparse.Namespace 或 dict。"""
    g = _getter(cfg)

    def _f(key, default):
        v = g(key, default)
        return float(default if v is None else v)

    bundle.reward_mode = str(g("reward_mode", "saturating") or "saturating")
    bundle.reward_vel_scale = _f("reward_vel_scale", 1.0)
    # 终止头（soft_termination 启发式）在新 WM 上误触发（survival 0.25–0.49），
    # 会把 20 步回报砍一半。`--term-off` 关掉它（survival≡1），真摔倒仍由漂移项惩罚。
    bundle.no_termination = 1.0 if bool(g("term_off", False)) else 0.0
    bundle.huber_delta = _f("huber_delta", 0.02)
    bundle.huber_cap = _f("huber_cap", 0.15)
    bundle.huber_err_d_max = _f("reward_d_max", 0.0)
    bundle.huber_err_delta_pos = _f("reward_delta_pos", 0.05)
    # 速度项的 Huber δ（`huber_err` 分支原来完全没有速度项，2026-11-05 才加）
    bundle.huber_err_delta_vel = _f("reward_delta_vel", 0.05)
    bundle.huber_err_delta_ori = _f("reward_delta_ori", 0.05)
    bundle.root_ori_mode = str(g("root_ori_mode", "split") or "split")
    bundle.tail_weight = _f("tail_weight", 2.0)
    bundle.tail_cap_slope = _f("tail_cap_slope", 0.0)
    bundle.yaw_hinge_lambda = _f("yaw_hinge_lambda", 0.3)
    _tpt = g("tail_pos_thr", None)
    _tpc = g("tail_pos_cap", None)
    if _tpt is not None:
        bundle.tail_pos_thr = float(_tpt)
    if _tpc is not None:
        bundle.tail_pos_cap = float(_tpc)
    _wyr = g("yawrate_weight", None)
    if _wyr is not None and float(_wyr) >= 0.0:
        bundle.yawrate_weight = float(_wyr)


def reward_cfg_summary(bundle) -> str:
    """一行审计串（训练与补算两侧打印同一份，便于对拍）。"""
    return (
        f"mode={bundle.reward_mode} vel_scale={bundle.reward_vel_scale:g} "
        f"huber_delta={bundle.huber_delta:g} huber_cap={bundle.huber_cap:g} "
        f"d_max={getattr(bundle, 'huber_err_d_max', 0.0):g} "
        f"d_pos={getattr(bundle, 'huber_err_delta_pos', 0.0):g} "
        f"d_vel={getattr(bundle, 'huber_err_delta_vel', 0.0):g} "
        f"d_ori={getattr(bundle, 'huber_err_delta_ori', 0.0):g} "
        f"root_ori_mode={bundle.root_ori_mode} tail_w={getattr(bundle, 'tail_weight', 0.0):g} "
        f"tail_cap_slope={getattr(bundle, 'tail_cap_slope', 0.0):g} "
        f"yaw_hinge_lambda={getattr(bundle, 'yaw_hinge_lambda', 0.0):g} "
        f"term_off={getattr(bundle, 'no_termination', 0.0):g}"
    )
