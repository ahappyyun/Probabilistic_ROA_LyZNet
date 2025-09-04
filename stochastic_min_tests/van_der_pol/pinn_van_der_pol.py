import sympy 
import lyznet
import numpy as np
import torch

lyznet.utils.set_random_seed()

# Define dynamics
mu = 1.0
# mu = 3.0
alpha, beta = 0.5, 0.5
x1, x2 = sympy.symbols('x1 x2')
f_vdp = [-x2, x1 - mu * (1 - x1**2) * x2]
sigma_vdp = [[alpha * x1, 0], [0, beta * x2]]
domain_vdp = [[-3.5, 3.5], [-4, 4]]
# domain_vdp = [[-5, 5], [-6, 6]]
sys_name = f"pinn_stoch_van_der_pol_alpha{alpha}_beta{beta}_mu_{mu}.py"

vdp_system = lyznet.StochasticDynamicalSystem(f_vdp, sigma_vdp, domain_vdp, sys_name)

print("System dynamics: x' = ", vdp_system.symbolic_f, "+", vdp_system.symbolic_sigma)
print("Domain: ", vdp_system.domain)

c1_P = lyznet.stochastic_local_stability_verifier(vdp_system)
c2_P = lyznet.stochastic_quadratic_reach_verifier(vdp_system, c1_P)

def drift(x, t):
    x1 = x[..., 0]
    x2 = x[..., 1]
    term1 = -x2
    term2 = x1 - mu * (1 - x1**2) * x2
    return torch.stack([term1, term2], dim=-1)

def diffusion(x, t):
    x1 = x[..., 0]
    x2 = x[..., 1]
    #  [alpha * x1, 0]
    #  [0, beta * x2]
    row1 = torch.stack([alpha * x1, torch.zeros_like(x1)], dim=-1)
    row2 = torch.stack([torch.zeros_like(x2), beta * x2], dim=-1)
    return torch.stack([row1, row2], dim=-2)


data = lyznet.generate_data_sde(vdp_system, drift, diffusion, n_samples=7000, num_simulations = 100)


# # Call the neural lyapunov learner
net, model_path = lyznet.neural_learner(vdp_system, lr=0.001, data=data, 
                                        layer=2, width=10, 
                                        num_colloc_pts=700000, max_epoch=8,
                                        loss_mode="sto_Zubov")

# Call the neural lyapunov verifier
c1_V, c2_V, W_1, C_1 = lyznet.stochastic_neural_verifier(vdp_system, net, c2_P)

lyznet.plot_V(vdp_system, net, model_path, c2_V=c2_V, c2_P=c2_P, phase_portrait=None)
# lyznet.plot_V_reproduce(vdp_system, net, model_path, c2_V=c2_V, c2_P=c2_P,
#               phase_portrait=None)
