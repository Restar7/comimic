# FM gradient ablation — scale (isolated)

K=[4], device=cuda

| mode | K | backend | loss | cosine vs pathwise | rel err | peak mem (MB) | time (s) | train loss before → after |
|---|---|---|---|---|---|---|---|---|
| scale | 4 | adjoint | 2339.7095 | +1.0000 | 0.00e+00 | 633.6 | 0.089 | 2339.709 → 167540.156 |
| scale | 4 | checkpointed | 2339.7095 | +1.0000 | 0.00e+00 | 448.8 | 0.106 | 2339.709 → 167540.156 |
| scale | 4 | pathwise | 2339.7095 | +1.0000 | 0.00e+00 | 575.6 | 0.062 | 2339.709 → 167540.156 |
| scale | 4 | truncated | 2339.7095 | +0.8869 | 7.57e-01 | 390.0 | 0.029 | 2339.709 → 89368.859 |
