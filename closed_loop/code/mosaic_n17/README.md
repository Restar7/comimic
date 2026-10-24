# mosaic_n17 — MOSAIC mechanisms on Isaac-GR00T N1.7 (SONIC78)

Additive package: **no upstream `gr00t/` file is modified.**  It implements the
migration specification in
`FlowBody/docs/projects/mosaic-n17-migration-plan-20261022.md` and the stage plan
in the FlowBody project record `docs/projects/isaac-groot-sonic-vla-finetune.md`.

## What is implemented

| Plan stage | Module | Status |
| --- | --- | --- |
| M0 baseline fine-tune | — (stock `examples/finetune.sh`) | **blocked**: gated `nvidia/Cosmos-Reason2-2B` returns 403 |
| M1 K-candidate sampling | `policy/candidate_sampler.py`, `policy/decode.py` | implemented, unit-tested with a stub head |
| M2 observation adapter / Phase A | `contracts/state46.py`, `ac/phase_a_runtime.py`, `data/{slots_h5,aux_targets}.py` | implemented; G1-ego cross-domain gate still to be run |
| M3 typed Phase B | `ac/{slots,phase_b_runtime,checkpoint}.py`, `train/train_phase_b.py` | implemented, CPU end-to-end test |
| M4 decision gate + shadow | `policy/{decision_gate,shadow}.py`, `policy/policy_wrapper.py` | implemented, unit-tested |
| M5 post-training | `posttrain/{teacher_bank,materialize,launch}.py`, `data/teacher_lerobot.py`, `eval/physics_replay.py` | implemented, tested (teacher dataset is loader-readable) |
| M6 VLM×world-model | `train/train_world_model_vlm.py` (`AuxSlotHead`) | head + entry implemented; **running needs the gated checkpoint** |
| M7 stage frontier / concordance | `stage/{frontier,concordance}.py` | implemented; needs a Stage-JEPA checkpoint configured for the typed-slot readout |

## Invariants enforced by code

- **No RTC on 78D tokens**: `sample_from_features` rejects RTC options; a token
  prefix has no established physical meaning.
- **Executed prefix only**: `validate_consequence_supervision` rejects targets
  beyond `execute_steps` (20 of 40), because the tail is never observed.
- **Verified teachers only**: the bank refuses `provisional` slots for training and
  `ProvenanceRecord.is_sim_verified` requires a simulation receipt digest.
- **No silent adapters**: N1.7 `state46` (43 joints + gravity) is converted to the
  AC layout (root RPY + leg/waist + arms + hands) by name-based tables; gravity is
  never used as root attitude.
- **Fail closed**: out-of-bound tokens, non-finite values, partially masked motion
  rows, wrong horizons and digest mismatches raise instead of being clipped.

## Layout

```
mosaic_n17/
  contracts/   sonic78.py  state46.py  timebase.py  provenance.py
  policy/      candidate_sampler.py  decode.py  decision_gate.py  shadow.py  policy_wrapper.py
  ac/          slots.py  phase_a_runtime.py  phase_b_runtime.py  scoring.py  checkpoint.py
  stage/       frontier.py  concordance.py
  data/        slots_h5.py  aux_targets.py  teacher_lerobot.py
  posttrain/   teacher_bank.py  materialize.py  launch.py
  eval/        oracle.py  calibration.py  physics_replay.py
  train/       train_phase_b.py  train_world_model_vlm.py
  configs/     k_ode.yaml  posttrain_teacher78.yaml
  tests/       72 offline tests (CPU)
```

## Running the tests

```bash
cd Isaac-GR00T
.venv/bin/python -m pytest mosaic_n17/tests -q
# cross-repo joint-order parity (optional):
FLOWBODY_ROOT=/path/to/FlowBody .venv/bin/python -m pytest mosaic_n17/tests -q
```

## Typical usage

```python
# 1) K candidates for one observation (model-ready batch from the processor)
from mosaic_n17.policy import sample_candidates, decide, run_shadow_step
candidates = sample_candidates(model, processor, inputs, K=4)
decision = decide(candidates.candidates[0], components=components)

# 2) teacher materialisation (verified slots only)
from mosaic_n17.posttrain import TeacherBank, materialize_samples
from mosaic_n17.data.teacher_lerobot import write_teacher_dataset
samples = materialize_samples(bank, source_rows, generator_sha256=..., ...)
write_teacher_dataset("runs/mosaic_n17/teachers/v1", samples)

# 3) post-training (demo + teacher mixing, no loss change)
from mosaic_n17.posttrain import load_posttrain_config, launch
config = load_posttrain_config("mosaic_n17/configs/posttrain_teacher78.yaml")
launch("mosaic_n17/configs/posttrain_teacher78.yaml", dry_run=True)
```

## Known blockers

1. `nvidia/Cosmos-Reason2-2B` is gated; without access no model can be loaded, so
   M0/M1-real/phase-B-real/M6 cannot run yet.
2. Phase A must first pass the G1-ego cross-domain probe (M2 exit criterion); the
   adapter refuses to construct without an explicit FlowBody checkpoint.
3. `torchcodec` 0.8 has no video encoder and torchvision needs PyAV, so teacher
   videos are encoded through the pinned `ffmpeg` CLI (H.264 / yuv420p).
