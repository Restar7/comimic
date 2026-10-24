# FM gradient ablation — tiny

K=[4], train_steps=20, lr=0.01, device=cuda, qvgm_critic=oracle

| mode | K | backend | loss | cosine vs pathwise | rel err | peak mem (MB) | time (s) | train loss before → after |
|---|---|---|---|---|---|---|---|---|
| tiny | 4 | pathwise | 724.3280 | +1.0000 | 0.00e+00 | 18.7 | 0.018 | 724.328 → 15.573 |
| tiny | 4 | checkpointed | 724.3280 | +1.0000 | 0.00e+00 | 19.5 | 0.033 | 724.328 → 15.573 |
| tiny | 4 | truncated | 724.3280 | +0.9975 | 7.49e-01 | 19.2 | 0.009 | 724.328 → 31.928 |
| tiny | 4 | adjoint | 724.3280 | +1.0000 | 0.00e+00 | 19.8 | 0.026 | 724.328 → 15.573 |
| tiny | 4 | qvgm_lite | 724.3152 | +0.9973 | 5.01e-01 | 19.2 | 0.009 | 724.328 → 100.120 |
