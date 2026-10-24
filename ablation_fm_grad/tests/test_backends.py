"""Backend correctness tests: all exact backends must agree with pathwise BPTT."""

from __future__ import annotations

import torch

from ablation_fm_grad.backends import make_backend
from ablation_fm_grad.harness import make_toy_setup, train_critic


def _grads(seed: int = 0, steps: int = 4):
    torch.manual_seed(seed)
    setup = make_toy_setup(steps, torch.device("cpu"), batch=4, dim=32, seed=seed)
    out = {}
    for name in ("pathwise", "checkpointed", "adjoint"):
        out[name] = make_backend(name).run(setup.field, setup.a0, steps, setup.task)
    return setup, out


def test_exact_backends_match_pathwise():
    setup, out = _grads()
    ref = out["pathwise"].grads
    for name in ("checkpointed", "adjoint"):
        other = out[name].grads
        for key in ref:
            torch.testing.assert_close(other[key], ref[key], rtol=2e-4, atol=2e-6)
        assert abs(out[name].loss - out["pathwise"].loss) < 1e-5


def test_finite_difference_of_pathwise():
    torch.manual_seed(0)
    setup = make_toy_setup(3, torch.device("cpu"), batch=2, dim=8, seed=0)
    backend = make_backend("pathwise")
    res = backend.run(setup.field, setup.a0, 3, setup.task)
    # pick the largest-gradient parameter entry
    name, grad = max(res.grads.items(), key=lambda kv: float(kv[1].abs().max()))
    param = dict(setup.field.named_parameters())[name]
    idx = grad.abs().argmax().item()
    flat = param.detach().reshape(-1)
    eps = 1e-3
    orig = flat[idx].item()
    flat[idx] = orig + eps
    loss_plus = float(setup.task(_sample(setup, 3)).detach())
    flat[idx] = orig - eps
    loss_minus = float(setup.task(_sample(setup, 3)).detach())
    flat[idx] = orig
    fd = (loss_plus - loss_minus) / (2 * eps)
    analytic = float(grad.reshape(-1)[idx])
    assert abs(fd - analytic) <= 2e-2 * max(1.0, abs(analytic)), (fd, analytic)


def _sample(setup, steps: int):
    with torch.no_grad():
        a = setup.a0
        for k in range(steps):
            a = a + (1.0 / steps) * setup.field(a, k)
    return a


def test_truncated_and_qvgm_are_finite():
    torch.manual_seed(0)
    setup = make_toy_setup(4, torch.device("cpu"), batch=4, dim=16, seed=0)
    tr = make_backend("truncated", tail=1).run(setup.field, setup.a0, 4, setup.task)
    assert all(torch.isfinite(g).all() for g in tr.grads.values())
    critic = train_critic(setup, 4, make_backend("pathwise"), iters=30)
    qv = make_backend("qvgm_lite").run(setup.field, setup.a0, 4, setup.task, critic=critic)
    assert all(torch.isfinite(g).all() for g in qv.grads.values())
