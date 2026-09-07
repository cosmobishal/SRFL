import numpy as np

from srfl import SRFLConfig, SRFLSolver


x = np.linspace(-np.pi, np.pi, 512)
target = (x >= 0).astype(float)
config = SRFLConfig(scales=64, lam_max=1.0, lam_min=0.02, dt=0.4, n_agents=8)
result = SRFLSolver(config).fit(x, target)

print(f"initial RMSE  {result.initial_error:.6f}")
print(f"final RMSE    {result.final_error:.6f}")
print(f"improvement   {result.improvement:.2%}")
print(f"events        {len(result.events)}")
