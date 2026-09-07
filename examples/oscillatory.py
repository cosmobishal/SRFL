import numpy as np

from srfl import SRFLConfig, SRFLSolver


x = np.linspace(-1.0, 1.0, 2048)
target = np.zeros_like(x)
mask = x != 0
target[mask] = x[mask] * np.sin(1.0 / x[mask])
config = SRFLConfig(scales=72, lam_max=0.35, lam_min=0.008, dt=0.35, n_agents=10,
                    detector_sensitivity=3.5, max_defects=8)
result = SRFLSolver(config).fit(x, target)

print(f"initial RMSE  {result.initial_error:.6f}")
print(f"final RMSE    {result.final_error:.6f}")
print(f"improvement   {result.improvement:.2%}")
print(f"max agents    {int(result.agent_counts.max())}")
