# Hinge recalibration for the 5-round full-length curriculum (v2, 2026-11-01)

Status: **v2 calibrated on real B_901 data; mid-field-gradient gates PASS.**  
This supersedes both the old R1 `e_h=0.35 / λ=1.5 / cap=1.0` table and the v1
`e_h=p75` table.

## Why v1 was wrong

v1 matched the old R1's **active fraction** (≈25%) by setting `e_h=p75`
(R1 `e_h=0.738`).  That looked statistically consistent but created a
**mid-field gradient dead zone**:

- root exp kernel σ=0.2 has value ≈1e-4 at 0.6 m and ≈5e-6 at 0.7 m;
- v1 hinge was exactly zero below 0.738 m;
- therefore a policy could stand still at ~0.5–0.7 m with **neither exp nor
  hinge gradient**.  That is exactly the standing failure observed in R1-fix.

Matching an old active fraction is not sufficient.  Calibration must also
guarantee a learning gradient in the common mid-field errors.

## v2 data and method

- Data: `runs/B_901/*/dump/*.pt` (180 full-length episodes, base VLA).
- Error labels: `runs/wm_real/self_targets_all.pt` (`e_inst` per tick).
- Start selection exactly mirrors training:
  - `t >= 10`, `stride=20`, `split=train` (drop every 5th replan);
  - keep `e_inst[t] <= E_start`;
  - require `t + H <= T`.
- For each selected start, take the **actual next H=20 ticks** from the real
  B rollout (not WM imagination) and pool their root errors.
- v2 parameters:
  - `e_h = p45` of the pooled error distribution ⇒ **active ≈55%**;
  - `cap = p95` (with `cap >= e_h + 0.10`);
  - `λ_h` solved so `mean(λ·hinge_raw) / mean(exp) = 0.30`;
  - `TAIL_CAP_SLOPE = 0.5` unchanged.
- The old R1 anchor (`E=0.5, e_h=.35, λ=1.5, cap=1.0`) is still reported for
  reference only; v2 deliberately uses a denser tail than its 0.123 ratio.
- Calibration script: `scripts/calibrate_hinge_5round.py`  
  Machine-readable result: `runs/S1_handover/hinge_calibration_5rounds.json`

```bash
python3 scripts/calibrate_hinge_5round.py \
  --out runs/S1_handover/hinge_calibration_5rounds.json
```

## Calibrated table (v2)

| 轮 | `E_start` | `e_h` | `λ_h` | `cap` | starts | e p45 | e p50 | e p90 | active | cap sat | grad@p50 | grad@p90 | hinge/exp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **R1** | **1.0** | **0.401** | **0.3592** | **0.971** | 2986 | 0.401 | 0.451 | 0.918 | 55.0% | 5.0% | 0.359 | 0.359 | 0.300 |
| **R2** | **1.2** | **0.485** | **0.2474** | **1.165** | 3548 | 0.485 | 0.550 | 1.094 | 55.0% | 5.0% | 0.247 | 0.247 | 0.300 |
| **R3** | **1.4** | **0.564** | **0.1931** | **1.327** | 4020 | 0.564 | 0.648 | 1.246 | 55.0% | 4.9% | 0.193 | 0.193 | 0.300 |
| **R4** | **1.6** | **0.632** | **0.1645** | **1.490** | 4373 | 0.632 | 0.719 | 1.381 | 55.0% | 5.0% | 0.165 | 0.165 | 0.300 |
| **R5** | **1.8** | **0.679** | **0.1441** | **1.637** | 4632 | 0.679 | 0.766 | 1.496 | 55.0% | 4.9% | 0.144 | 0.144 | 0.300 |

Reference only (old R1):

| `E_start` | `e_h` | `λ_h` | `cap` | active | hinge/exp |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.35 | 1.5 | 1.0 | 30.6% | 0.1232 |

## Calibration gates (all PASS)

- active fraction in `[0.45, 0.65]` for every round;
- `grad@p50 > 0` and `grad@p90 > 0` for every round;
- cap-saturated fraction `<= 0.08` for every round;
- `e_h < E_start` and `e_h < cap` for every round;
- `e_h` strictly increases R1→R5;
- `λ_h` strictly decreases R1→R5;
- `mean(hinge)/mean(exp) = 0.300` for every round.

## Scope and caveats

- This calibrates the **root-position hinge** (`ROOT_POS_THR`, `ROOT_POS_CAP`,
  `TAIL_WEIGHT`).  The tilt hinge remains at fixed `20°/57°`; yaw hinge stays
  disabled (`yaw_hinge=0`).
- Calibration uses **B_901** only.  When R2–R5 collect new B data, re-run the
  same script on the cumulative pool and re-lock the table before joint training.
- `B_SELF_ERR_MAX` stays **empty**: future-error filtering still biases starts
  to early phases.  Start difficulty is controlled by `E_start`.
- Values must be consumed from the JSON/table, not the old constants.

## v3 λ principle (2026-11-02): match R1-fix4 at p90

R1-fix4 (the first checkpoint that passed R1) used `e_h=0.401`, `λ=1.0`,
`cap=0.971`; its H=20 error p90 was **0.918 m**, so its p90 hinge penalty was

```text
1.0 × (0.918 − 0.401) = 0.517
```

For each later round, `e_h`/`cap` stay at the v2 `p45/p95` values, and λ is chosen
so the **absolute p90 hinge penalty also equals 0.517**:

```text
λ_round = 0.517 / (p90_round − e_h_round)     (p90 below cap)
```

| round | e_h | error p90 | p90 tail | λ = 0.517/tail | rounded |
|---|---:|---:|---:|---:|---:|
| R1-fix4 | 0.401 | 0.918 | 0.517 | 1.000 | 1.0 |
| R2 | 0.485 | 1.094 | 0.609 | 0.849 | **0.85**（R2 已用早期 0.247 跑完，保留实际） |
| R3 | 0.564 | 1.246 | 0.682 | 0.758 | **0.80** |
| R4 | 0.632 | 1.381 | 0.749 | 0.690 | **0.70** |
| R5 | 0.679 | 1.496 | 0.817 | 0.633 | **0.65** |

Why not copy λ=1.0 blindly: R3–R5 have larger p90 errors; 1.0 would make the
far tail 32–42% stronger than R1 and can suppress reward.  Why not the old v2
0.19/0.16/0.14: those are only 25–30% of R1's p90 penalty and leave the far
field under-constrained (R3 `imag_state` degraded to ~1.54).

R2 was already trained with λ=0.2474; R3+ use the p90-matched values above.
