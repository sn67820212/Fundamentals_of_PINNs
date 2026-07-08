# Section B: extension of pde to solve 2 ddimensional problems.
# The network takes 2 inputs (x,y) and has one scalar output u(x,y).
# Autograd computes partial derivatives naturally, just with two input columns.
# The collocation points is replced by 2D mesh grid.
# Boundary condition appear on all 4 edges.

import torch
import torch.nn as nn

# Level 3: 2D Laplacian: u_xx + u_yy = 0 
# BCs: zero at 3 layers except top where u(x,1)=sin(pi*x)
class PINN2D:
    def __init__(self,layers,lr=1e-3): 
        net = []
        for i in range(len(layers)-1): 
            net.append(nn.Linear(layers[i],layers[i+1]))
            if i < len(layers)-2 :
                net.append(nn.Tanh())
        self.model = nn.Sequential(*net)
        self.optimizer = torch.optim.Adam (self.model.parameters(),lr=lr)
    
    def forward (self,xy):
        return self.model(xy)
    
    def loss_pde (self,xy):
        u = self.forward(xy)
        u_x = torch.autograd.grad(u.sum(),xy,create_graph=True)[0][:,0:1]
        u_y = torch.autograd.grad(u.sum(),xy,create_graph=True)[0][:,1:2]
        u_xx = torch.autograd.grad(u.sum(),xy,create_graph=True)[0][:,0:1]
        u_yy: torch.Tensor = torch.autograd.grad(u.sum(),xy,create_graph=True)[0][:,1:2]
        return (u_xx+u_yy).pow(2).mean()
    
    def loss_bc (self,xy_b,u_b):
        return (self.forward(xy_b)-u_b).pow(2).mean()
    
    def train_step (self,xy,xy_b,u_b):
        loss = self.loss_pde(xy) + 10*self.loss_bc(xy_b,u_b)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # return individual losses for dignosis
        return loss.item()

# 2D collocation grid: 30x30 interior points
N = 30
x_vals = torch.linspace(0,1,N)
y_vals = torch.linspace(0,1,N)
X, Y = torch.meshgrid (x_vals, y_vals, indexing= 'ij')
xy = torch.stack ([X.flatten(), Y.flatten()],dim=1)
xy = xy.requires_grad_(True)

# collocations on Boundary: u=0 on three sides, u=sin(pi*x) on top
import math
n_b = 50
y_b = x_b = torch.linspace (0,1,n_b)
# bottom: y=0, u=0
xy_bot = torch.stack ([x_b,torch.zeros(n_b)], dim=1)
u_bot = torch.zeros (n_b, 1)
# top: y=1, u= sin (pi*x)
xy_top = torch.stack ([x_b, torch.ones(n_b)],dim=1) # shape of xy_top = 50,2
u_top = torch.sin (math.pi*x_b).reshape(-1,1) #shape of u_top = 50,1
# left: x=0, u=0
xy_lft = torch.stack ([torch.zeros(n_b), y_b], dim=1)
u_lft = torch.zeros (n_b, 1)
# right: x=1, u=0
xy_rgt = torch.stack ([torch.ones(n_b), y_b], dim=1)
u_rgt = torch.zeros (n_b, 1)

xy_b = torch.cat ([xy_bot,xy_top,xy_lft,xy_rgt], dim=0)
u_b = torch.cat ([u_bot,u_top,u_lft,u_rgt],dim=0)
pinn = PINN2D(layers=[2,30,30,30,1])
for step in range(5000):
    loss = pinn.train_step(xy,xy_b,u_b)
    if step % 500 == 0:
        print (f"step {step}: loss = {loss}")
# The loss was converging slowly than one dimensional case as:
# there are two parameters and collocation points is replaced by 2D grid.