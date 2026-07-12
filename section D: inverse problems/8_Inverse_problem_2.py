# Level 8: recovering a varying coefficient
"""
Problem: -d/dx (k(x) du/dx) = f(x), source term [f(x)] = 1 (uniform source)
where k(x) unknown spatially varying coefficient e.g. thermal conductivity, permeability
when part of u(x) is known usually by experiment, we aim to find k(x) by using PINN.
true k(x) = 1 + 0.5*sin(2*pi*x)
"""

import torch
import torch.nn as nn

class DualNetPINn:
    def __init__(self, lr=1e-3):
        self.u_net = nn.Sequential(
            nn.Linear(1,30), nn.Tanh(),
            nn.Linear(30,30),nn.Tanh(),
            nn.Linear(30,1))
        self.k_net = nn.Sequential(
            nn.Linear(1,20), nn.Tanh(),
            nn.Linear(20,20),nn.Tanh(),
            nn.Linear(20,1),nn.Softplus())
        # softplus ensures k(x) > 0
        self.optimizer = torch.optim.Adam(
            list(self.u_net.parameters())+
            list(self.k_net.parameters()),
            lr = lr)
    
    def loss_pde(self,x):
        u = self.u_net(x)
        k = self.k_net(x)
        u_x = torch.autograd.grad(u.sum(),x,create_graph=True)[0]
        ku_x = k*u_x
        dku_x = torch.autograd.grad(
            ku_x.sum(), x, create_graph=True)[0]
        return (dku_x+1).pow(2).mean()
    
    def loss_data(self, x_data,u_data):
        return (self.u_net(x_data)-u_data).pow(2).mean()
    
    def train_step(self,x,x_data,u_data):
        loss = self.loss_pde(x) + 20*self.loss_data(x_data,u_data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
import math

# generate data using, u = x(1-x)/2
# true k(x) = 1 + 0.5*sin(2*pi*x)
x_data = torch.linspace(0,1,100).reshape(-1,1)
u_data = x_data*(1-x_data)/2
x = torch.linspace(0,1,200).reshape(-1,1).requires_grad_(True)

pinn = DualNetPINn()
for step in range(5000):
    loss = pinn.train_step(x,x_data,u_data)
    if step % 1000 == 0:
        k_half = pinn.k_net
        print(f"step {step}: loss={loss}, k(0.5)={k_half}")

import matplotlib.pyplot as plt
true_k = 1 + 0.5*torch.sin(2*math.pi*x)
k_guess = pinn.k_net(x)
true_k = true_k.detach().numpy()
k_guess = k_guess.detach().numpy()
x=x.detach().numpy()
plt.plot (x,true_k,label='exact solution')
plt.plot (x,k_guess,label='PINN_solution')
plt.legend()
plt.show()

# my model got settled at x=1.
# upon further investigation, I found k(x)=1 also satisfies the given problem.
# So, the model tends to learn simple solution.
# This is one of the limitation of PINN.
# U get only one model and there can be multiple solutions which u wouldn't know with nn.