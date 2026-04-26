"""
run_tests.py — Standalone SRFL test runner (no pytest required).
Runs all test classes via the built-in test registry and prints a full
pass/fail report.

Usage
-----
    python run_tests.py
"""
import sys
import os

# Ensure src/ is on the path when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Delegate to the full test suite
import traceback, time
import numpy as np

from srfl import (
    SRFLKernel, SRFLField, Swarm,
    StepDefect, OscillatoryDefect, ConditionalDefect, DefectAlgebra,
    ActionFunctional, ScaleProjection,
)
from srfl.field import SingularityGenerator

# ── shared fixtures ───────────────────────────────────────────────────────────
N_SMALL   = 256
x_s       = np.linspace(-np.pi, np.pi, N_SMALL)
dx_s      = float(x_s[1] - x_s[0])
lam_s     = np.logspace(0, -1.5, 30)
target_s  = np.where(x_s >= 0, 1.0, 0.0).astype(float)


def _osc(x):
    out = np.zeros_like(x); nz = x != 0
    out[nz] = x[nz] * np.sin(1.0 / x[nz])
    return out


# ── test registry ─────────────────────────────────────────────────────────────
PASS = []; FAIL = []; ERRORS = []


def test(name):
    def decorator(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  ✓  {name}")
        except AssertionError as e:
            FAIL.append((name, str(e)))
            print(f"  ✗  {name}  →  AssertionError: {e}")
        except Exception as e:
            ERRORS.append((name, traceback.format_exc()))
            print(f"  !  {name}  →  {type(e).__name__}: {e}")
        return fn
    return decorator


def approx_eq(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  SRFL TEST SUITE")
print("="*60)

# ── SRFLKernel ───────────────────────────────────────────────────────────────
print("\n[ SRFLKernel ]")

@test("kernel: positive lambda ok")
def _(): assert SRFLKernel(x_s, 0.5).lam == 0.5

@test("kernel: negative lambda raises ValueError")
def _():
    try: SRFLKernel(x_s, -0.1); assert False
    except ValueError: pass

@test("kernel: zero lambda raises ValueError")
def _():
    try: SRFLKernel(x_s, 0.0); assert False
    except ValueError: pass

@test("kernel: convolve preserves shape")
def _():
    K = SRFLKernel(x_s, 0.3)
    assert K.convolve(np.random.randn(N_SMALL)).shape == (N_SMALL,)

@test("kernel: convolve smooths high-frequency input")
def _():
    K = SRFLKernel(x_s, 0.5)
    phi = np.sin(50 * x_s)
    assert np.std(K.convolve(phi)) < np.std(phi)

@test("kernel: convolve(constant) ≈ constant")
def _():
    K   = SRFLKernel(x_s, 0.4)
    phi = np.ones(N_SMALL) * 3.14
    assert np.allclose(K.convolve(phi), phi, atol=1e-3)

@test("kernel: approximate identity as λ→0")
def _():
    K   = SRFLKernel(x_s, 1e-3)
    phi = np.sin(2 * x_s)
    assert np.sqrt(np.mean((K.convolve(phi) - phi)**2)) < 0.01

@test("kernel: semigroup K_λ * K_μ = K_{√(λ²+μ²)}")
def _():
    lam, mu = 0.3, 0.4
    K1, K2  = SRFLKernel(x_s, lam), SRFLKernel(x_s, mu)
    K3      = SRFLKernel(x_s, np.sqrt(lam**2 + mu**2))
    phi     = np.sin(x_s) + 0.5 * np.cos(3*x_s)
    err     = np.sqrt(np.mean((K2.convolve(K1.convolve(phi)) - K3.convolve(phi))**2))
    assert err < 1e-5, f"semigroup err={err:.2e}"

@test("kernel: FWHM = 2√(2ln2)·λ")
def _():
    lam = 0.6; K = SRFLKernel(x_s, lam)
    expected = 2.0 * np.sqrt(2.0 * np.log(2.0)) * lam
    assert approx_eq(K.fwhm(), expected)

@test("kernel: matrix shape")
def _():
    K = SRFLKernel(x_s, 0.5); M = K.matrix(subsample=4)
    assert M.shape == (N_SMALL // 4, N_SMALL // 4)

@test("kernel: agent_kernel self = 1.0")
def _():
    K = SRFLKernel(x_s, 0.3)
    assert approx_eq(K.agent_kernel(0.5, 0.5, 0.3, 0.3), 1.0)

@test("kernel: agent_kernel decreases with distance")
def _():
    K = SRFLKernel(x_s, 0.5)
    assert K.agent_kernel(0.0, 0.1) > K.agent_kernel(0.0, 1.0)

@test("kernel: update_lambda immutable")
def _():
    K1 = SRFLKernel(x_s, 0.5); K2 = K1.update_lambda(0.1)
    assert K2.lam == 0.1 and K1.lam == 0.5

@test("kernel: repr contains SRFLKernel")
def _(): assert "SRFLKernel" in repr(SRFLKernel(x_s, 0.25))

# ── SingularityGenerator ─────────────────────────────────────────────────────
print("\n[ SingularityGenerator ]")

@test("singularity: output shape")
def _():
    S = SingularityGenerator(kappa=1.8, eps=0.07)
    assert S(np.sin(x_s), dx_s).shape == (N_SMALL,)

@test("singularity: zero for very smooth field (high κ)")
def _():
    S   = SingularityGenerator(kappa=10.0, eps=0.07)
    out = S(0.01 * np.sin(x_s), dx_s)
    assert np.allclose(out, 0.0)

@test("singularity: fires on steep field (low κ)")
def _():
    S   = SingularityGenerator(kappa=0.5, eps=0.1)
    phi = np.where(x_s >= 0, 1.0, 0.0).astype(float)
    assert np.any(S(phi, dx_s) != 0.0)

@test("singularity: output always finite")
def _():
    S   = SingularityGenerator(kappa=0.1, eps=0.5)
    out = S(np.random.randn(N_SMALL) * 10, dx_s)
    assert np.all(np.isfinite(out))

# ── SRFLField ────────────────────────────────────────────────────────────────
print("\n[ SRFLField ]")

@test("field: mismatched target length raises ValueError")
def _():
    try: SRFLField(x_s, np.ones(100), lam_s); assert False
    except ValueError: pass

@test("field: increasing lam_sched raises ValueError")
def _():
    try: SRFLField(x_s, target_s, np.linspace(0.01,1.0,20)); assert False
    except ValueError: pass

@test("field: run returns correct list lengths")
def _():
    engine = SRFLField(x_s, target_s, lam_s)
    f, e   = engine.run()
    assert len(f) == len(lam_s) and len(e) == len(lam_s)

@test("field: errors are non-negative")
def _():
    _, e = SRFLField(x_s, target_s, lam_s).run()
    assert all(err >= 0.0 for err in e)

@test("field: L² error strictly decreases coarse→fine")
def _():
    _, e = SRFLField(x_s, target_s, lam_s).run()
    assert e[-1] < e[0], f"err[-1]={e[-1]:.4f} >= err[0]={e[0]:.4f}"

@test("field: all field values within clip bound")
def _():
    clip = 2.8
    f, _ = SRFLField(x_s, target_s, lam_s, clip=clip).run()
    for phi in f:
        assert np.all(phi >= -clip - 1e-9) and np.all(phi <= clip + 1e-9)

@test("field: final field closer to target than initial")
def _():
    f, _ = SRFLField(x_s, target_s, lam_s).run()
    assert np.sqrt(np.mean((f[-1]-target_s)**2)) < np.sqrt(np.mean((f[0]-target_s)**2))

@test("field: repr contains SRFLField")
def _(): assert "SRFLField" in repr(SRFLField(x_s, target_s, lam_s))

# ── Defects ──────────────────────────────────────────────────────────────────
print("\n[ Defects ]")

@test("step_defect: field is Heaviside-shaped")
def _():
    D = StepDefect(x0=0.0, alpha=1.0); out = D.field(x_s)
    assert np.all(out[x_s < 0] == 0.0) and np.all(out[x_s >= 0] == 1.0)

@test("step_defect: apply is additive")
def _():
    D = StepDefect(x0=0.0, alpha=2.0)
    out = D.apply(x_s, np.ones(N_SMALL))
    assert np.allclose(out[x_s >= 0], 3.0)

@test("step_defect: norm = |α|")
def _(): assert StepDefect(x0=0.0, alpha=-3.5).norm() == 3.5

@test("step_defect: compose same x0 sums alpha")
def _():
    D = StepDefect(0.0, 1.0).compose(StepDefect(0.0, 2.0))
    assert D.alpha == 3.0

@test("osc_defect: |D(x)| ≤ |x| everywhere")
def _():
    D   = OscillatoryDefect(eps=0.5, beta=1.0)
    out = D.field(x_s)
    mask = np.abs(x_s) < 0.5
    assert np.all(np.abs(out[mask]) <= np.abs(x_s[mask]) + 1e-12)

@test("osc_defect: zero outside support")
def _():
    D   = OscillatoryDefect(eps=0.3, beta=1.0)
    out = D.field(x_s)
    assert np.all(out[np.abs(x_s) >= 0.3] == 0.0)

@test("cond_defect: piecewise-constant values")
def _():
    D   = ConditionalDefect([(-1.0, 0.0), (0.0, 1.0)], [2.0, -1.0])
    out = D.field(x_s)
    assert np.allclose(out[(x_s >= -1.0) & (x_s < 0.0)], 2.0)
    assert np.allclose(out[(x_s >= 0.0)  & (x_s < 1.0)], -1.0)

@test("cond_defect: norm = sum |aₖ|")
def _():
    assert ConditionalDefect([(-1,0),(0,1)], [3.0,-4.0]).norm() == 7.0

@test("defect_algebra: commutator of identical defect = 0")
def _():
    alg = DefectAlgebra(x_s)
    D   = StepDefect(0.0, 1.0)
    com = alg.commutator_field(D, D, np.zeros(N_SMALL))
    assert np.allclose(com, 0.0)

@test("defect_algebra: detect_from_curvature returns list")
def _():
    defs = DefectAlgebra.detect_from_curvature(
               x_s, np.where(x_s>=0,1.0,0.0).astype(float), dx_s)
    assert isinstance(defs, list)

# ── Swarm ────────────────────────────────────────────────────────────────────
print("\n[ Swarm ]")

@test("swarm: initial count correct")
def _(): assert Swarm(x_s, n_init=8).count() == 8

@test("swarm: single step runs without error")
def _(): Swarm(x_s, n_init=8).step(np.zeros(N_SMALL), target_s, 0.5, 1)

@test("swarm: count always ≥ 1 after many steps")
def _():
    sw = Swarm(x_s, n_init=8, spawn_period=5, merge_eps=0.15, annihil_thresh=0.05)
    for k in range(20):
        sw.step(np.zeros(N_SMALL), target_s, 0.5*(1-k/40), k)
    assert sw.count() >= 1

@test("swarm: all agent positions within domain")
def _():
    sw = Swarm(x_s, n_init=8)
    for k in range(10):
        sw.step(np.zeros(N_SMALL), target_s, 0.4, k)
    for p in sw.positions():
        assert x_s[0] <= p <= x_s[-1]

@test("swarm: interaction matrix diagonal = 1")
def _():
    sw = Swarm(x_s, n_init=6)
    assert np.allclose(np.diag(sw.interaction_matrix(0.5)), 1.0)

@test("swarm: event_summary has correct keys")
def _():
    assert set(Swarm(x_s).event_summary().keys()) == {"spawn","merge","annihilate"}

# ── ActionFunctional ─────────────────────────────────────────────────────────
print("\n[ ActionFunctional ]")

_af_fields, _ = SRFLField(x_s, target_s, lam_s, dt=0.25).run()
_af = ActionFunctional(x_s, lam_s, target_s)

@test("action: A_data non-negative")
def _(): assert _af.A_data(_af_fields) >= 0.0

@test("action: A_sym = 0 without symmetry operator")
def _(): assert _af.A_sym(_af_fields) == 0.0

@test("action: total() returns all five keys")
def _():
    res = _af.total(_af_fields)
    assert set(res.keys()) == {"data","scale","symmetry","complexity","total"}

@test("action: total == sum of parts")
def _():
    r = _af.total(_af_fields)
    manual = r["data"] + r["scale"] + r["symmetry"] + r["complexity"]
    assert approx_eq(r["total"], manual, tol=1e-10)

# ── ScaleProjection ───────────────────────────────────────────────────────────
print("\n[ ScaleProjection ]")

_proj     = ScaleProjection(x_s)
_phi_test = np.sin(x_s) + 0.3 * np.cos(3*x_s)

@test("proj: project same scale = identity")
def _():
    assert np.allclose(_proj.project(_phi_test, 0.5, 0.5), _phi_test)

@test("proj: semigroup property holds to 1e-5")
def _():
    err, ok = _proj.verify_semigroup(_phi_test, 0.2, 0.5, 0.8, tol=1e-5)
    assert ok, f"semigroup error {err:.2e}"

@test("proj: reverse direction raises ValueError")
def _():
    try: _proj.project(_phi_test, 0.8, 0.2); assert False
    except ValueError: pass

# ── Integration ────────────────────────────────────────────────────────────────
print("\n[ Integration ]")

@test("integration: full pipeline step function")
def _():
    x     = np.linspace(-np.pi, np.pi, 128)
    lams  = np.logspace(0, -1.5, 20)
    tgt   = np.where(x >= 0, 1.0, 0.0).astype(float)
    f, e  = SRFLField(x, tgt, lams, dt=0.25).run()
    sw    = Swarm(x, n_init=6)
    for k, (phi, lam) in enumerate(zip(f[1:], lams[1:]), 1):
        sw.step(phi, tgt, lam, k)
    A  = ActionFunctional(x, lams, tgt).total(f)
    assert e[-1] < e[0]
    assert A["total"] >= 0.0
    assert sw.count() >= 1

@test("integration: CLI run_single returns dict with expected keys")
def _():
    import argparse
    from srfl.cli import run_single
    args = argparse.Namespace(
        n=64, steps=10, lam0=1.0, lam1=0.1, dt=0.25,
        n_agents=4, outdir="/tmp", no_figures=True, verbose=False)
    result = run_single("sine", args)
    assert "fields" in result and "errors" in result

# ── Final report ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
total = len(PASS) + len(FAIL) + len(ERRORS)
print(f"  Results:  {len(PASS)} passed  |  {len(FAIL)} failed  |  {len(ERRORS)} errors  |  {total} total")
if FAIL:
    print("\n  FAILURES:")
    for name, msg in FAIL:
        print(f"    ✗ {name}: {msg}")
if ERRORS:
    print("\n  ERRORS:")
    for name, tb in ERRORS:
        print(f"    ! {name}:\n{tb}")
print("="*60)
sys.exit(0 if not FAIL and not ERRORS else 1)
