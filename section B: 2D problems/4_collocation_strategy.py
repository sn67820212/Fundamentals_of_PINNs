# Collocation strategy
# So far we discussed about uniform grid method to sample points in a region.
# Uniform grid works greatly for smooth solutions.
# near sharp gradients or boundaries, we need denser sampling.
# i.e. network should get more signals where physics changes rapidly.

import numpy as np
import torch

def uniform_collocation (n,dim =2):
    pts = torch.rand (n,dim)
    return pts.requires_grad_(True)

def boundary_collocation(n_per_edge):
    t = torch.linspace (0, 1, n_per_edge)
    z = torch.zeros (n_per_edge)
    o = torch.ones (n_per_edge)
    bot = torch.stack ([t,z], dim=1)
    top = torch.stack ([t,0], dim=1)
    lft = torch.stack ([z,t], dim=1)
    rgt = torch.stack ([o,t], dim=1)
    return torch.cat([bot, top, lft, rgt], dim=0)

def dense_near_boundary (n_interior, n_boundary_layer):
    interior = torch.rand (n_interior,2)
    near_bot = torch.rand (n_boundary_layer,2)
    near_bot[:,1] = near_bot [:, 1] * 0.05 # scale y coordinate by 0.05 which forces heavy collocation
    return torch.cat ([interior, near_bot], dim=0).requires_grad_(True)

# We can choose any of the sampling method based on the nature of problem and convenience.
# The physics and math should decide the sampling method itself, so we leave it here.
# I am ending collocation here as this is enough to begin steps in PINN without getting stuck which is better than:
# looping in same idea and getting thrown by too fast learning rate without covering basics.
# Next, we will discuss time dependent problem before entering inverse problems which is heart of PINN as far as I am concerned.