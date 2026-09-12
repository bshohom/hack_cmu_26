"""torch-fem's official 3D SIMP cantilever example (examples/optimization/solid/topology.ipynb),
headless. Run this first: if it works, the numeric stack is fine."""

import time

import torch
from scipy.optimize import bisect
from torchfem import Solid
from torchfem.materials import IsotropicElasticity3D
from torchfem.mesh import cube_hexa

torch.set_default_dtype(torch.float64)

material = IsotropicElasticity3D(E=100.0, nu=0.3)
Nx, Ny, Nz = 20, 10, 15
nodes, elements = cube_hexa(Nx + 1, Ny + 1, Nz + 1, Nx, Ny, Nz)
model = Solid(nodes, elements, material)

tip = nodes[:, 0] == Nx
bottom = nodes[:, 2] == 0
model.forces[tip & bottom, 2] = -1.0
model.forces[tip & bottom & (nodes[:, 1] == 0), 2] = -0.5
model.forces[tip & bottom & (nodes[:, 1] == Ny), 2] = -0.5
model.constraints[nodes[:, 0] == 0.0, :] = True

volfrac, p, move, R = 0.5, 3, 0.2, 1.5
rho = volfrac * torch.ones(len(elements))
rho_min, rho_max = 0.05 * torch.ones_like(rho), torch.ones_like(rho)
V_0 = volfrac * Nx * Ny * Nz
k0 = model.k0()
C0 = model.material.C.clone()
ecenters = nodes[elements].mean(dim=1)
dist = torch.cdist(ecenters, ecenters)
H = R - dist
H[dist > R] = 0.0

t0 = time.time()
for k in range(20):
    model.material.C = torch.einsum("n,nijkl->nijkl", rho**p, C0)
    u_k, f_k, _, _, _ = model.solve()
    compliance = torch.inner(f_k.ravel(), u_k.ravel())
    u_j = u_k[elements].reshape(model.n_elem, -1)
    w_k = torch.einsum("...i, ...ij, ...j", u_j, k0, u_j)
    sensitivity = -p * rho ** (p - 1.0) * w_k
    sensitivity = H @ (rho * sensitivity) / H.sum(dim=0) / rho

    def make_step(mu):
        G_k = -sensitivity / mu
        upper = torch.min(rho_max, (1 + move) * rho)
        lower = torch.max(rho_min, (1 - move) * rho)
        return torch.maximum(torch.minimum(G_k**0.5 * rho, upper), lower)

    def g(mu):
        return make_step(mu).sum() - V_0

    with torch.no_grad():
        mu = bisect(g, 1e-10, 100.0)
    rho = make_step(mu)
    print(f"iter {k:2d}  compliance {compliance.item():10.4f}  vol {rho.sum().item() / (Nx * Ny * Nz):.3f}")
print(f"done in {time.time() - t0:.1f} s")
