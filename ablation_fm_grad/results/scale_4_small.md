# FM gradient ablation — scale (isolated)

K=[4], device=cuda

| mode | K | backend | loss | cosine vs pathwise | rel err | peak mem (MB) | time (s) | train loss before → after |
|---|---|---|---|---|---|---|---|---|
| scale | 4 | adjoint | 2324.5945 | +1.0000 | 0.00e+00 | 110.4 | 0.056 | 2324.594 → 1625.294 |
| scale | 4 | checkpointed | 2324.5945 | +1.0000 | 0.00e+00 | 81.4 | 0.065 | 2324.594 → 1625.294 |
| scale | 4 | pathwise | 2324.5945 | +1.0000 | 0.00e+00 | 97.2 | 0.036 | 2324.594 → 1625.294 |
| scale | 4 | truncated | 2324.5945 | +0.9436 | 7.55e-01 | 73.2 | 0.018 | 2324.594 → 1787.190 |
