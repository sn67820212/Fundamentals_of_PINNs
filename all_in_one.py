import torch
import torch.nn as nn

# This part will show you a professional structre of PINN.
# The reader is expected to know few basic syntaxes of torch and class.

# Section A: Architecture of PINN ( reusable code structre not just solving problem)
# Level 1: create a class PINN that contains model, optimizer and training part 

# When we do actual work in physics. We do not want to type same loop each times.
# To remove that difficulty, we create a class with defined functions.
# We will use of classes, neral networks to automate the work.
# Hence, u should have at least basic knowledge of these libraries to understand this code.
# For class, any tutorial on any platform like youtube is enough. 
class PINN:
    def __init__(self,layers,lr=1e-3): # layers is mandatory entry, lr is 1e-3 by default and can be ajusted when calling the class. 
        # to build network automatically from list of layer widths
        # e.g.layers = [1,20,20,1] ::: 1 to 20 to 20 to 1. two input and output layers along with two hidden layers.
        net = [] # We follow this structre before introducing model specially if we have too many layers.
        # This structre is not mandatary if layers' number is fewer than 5 but by just changing the list layers, this:
        # reproduces another model with an optimizer and these automations are helpful when u need to be more professional.
        # u can define model and optimizer manually inside nn.Sequential if u want simple structre. 
        for i in range(len(layers)-1): # 4-1 = 3 steps transformation
            net.append(nn.Linear(layers[i],layers[i+1])) #1,20; 20,20; 20,1
            if i < len(layers)-2 : # last layer don't need nonlinearity
                net.append(nn.Tanh())
            self.model = nn.Sequential(*net)
            self.optimizer = torch.optim.Adam (self.model.parameters(),lr=lr)

    def forward(self,x):
        return self.model(x)
    def loss_pde (self,x):
        # u" + 1 = 0
        u = self.forward(x)
        u_x = torch.autograd.grad(u.sum(),x,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(),x,create_graph=True)[0]
        return (u_xx+1).pow(2).mean()
    
    def loss_bc (self,x_b,u_b):
        return (self.forward(x_b)-u_b).pow(2).mean()
    def train_step (self,x,x_b,u_b):
        loss = self.loss_pde(x) + self.loss_bc(x_b,u_b)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
# We have successfully created a class that has every characteristic a neural network shoul have.
# We can call loss, training or whatever we want by just giving BCs, layer size,step etc.
# Usage to showcase how to use class PINN
# define neural network by layer structe

#e.g. 1.1: layers=[1,20,20,1] , Boundary conditions(BCs): u(0)=u(1)=0
"""pinn = PINN(layers=[1,20,20,1])
# Defining a neural network by the class may seem long at first.
# But if u observe carefully, class PINN has general structre that can be used for any neural network with any layer structre and BCs. 
# You can recreate class PINN for other pde by just modifying loss_pde while keeping everything else same.
# collocation points
x = torch.linspace(0,1,100).reshape(-1,1).requires_grad_(True)
# boundary conditions
x_b = torch.tensor([[0.],[1.]]) # x belongs to [0,1]
u_b = torch.tensor([[0.],[0.]]) # u(0)=u(1)=0

# training the model 
for step in range (3000):
    loss = pinn.train_step(x, x_b, u_b)
    # sample losses for validation
    if step % 500 == 0:
        print (f"step {step}: {loss}")
with torch.no_grad():
    print(pinn.forward(torch.tensor([[0.5]])).item()) # 0.125"""

#e.g. 1.2: layers=[1,50,50,50,1] with same BC as 1.1, let's check which converge faster
"""pinn = PINN(layers=[1,50,50,50,1])

# collocation points
x = torch.linspace(0,1,100).reshape(-1,1).requires_grad_(True)
# boundary conditions
x_b = torch.tensor([[0.],[1.]]) # x belongs to [0,1]
u_b = torch.tensor([[0.],[0.]]) # u(0)=u(1)=0

# training the model 
for step in range (3000):
    loss = pinn.train_step(x, x_b, u_b)
    # sample losses for validation
    if step % 500 == 0:
        print (f"step {step}: {loss}")
with torch.no_grad():
    print(pinn.forward(torch.tensor([[0.5]])).item())""" # 0.125
# It should be obvious that larger hidden layer converges faster.

# Let's change lr to see which lr converges faster without overshooting
pinn = PINN(layers=[1,50,50,50,1],lr=10) 

# collocation points
x = torch.linspace(0,1,100).reshape(-1,1).requires_grad_(True)
# boundary conditions
x_b = torch.tensor([[0.],[1.]]) # x belongs to [0,1]
u_b = torch.tensor([[0.],[0.]]) # u(0)=u(1)=0

# training the model 
for step in range (3000):
    loss = pinn.train_step(x, x_b, u_b)
    # sample losses for validation
    if step % 500 == 0:
        print (f"step {step}: {loss}")
with torch.no_grad():
    print(pinn.forward(torch.tensor([[0.5]])).item()) 
# The error never converge to a certian value if u use large lr.
# Rather, it starts to diverge after certain point.
# The optimum lr from hit and trial was closer to 1e-4 to 1e-2.
import torch
import torch.nn as nn

# This part will show you a professional structre of PINN.
# The reader is expected to know few basic syntaxes of torch and class.

# Section A: Architecture of PINN ( reusable code structre not just solving problem)
# Level 1: create a class PINN that contains model, optimizer and training part 

# When we do actual work in physics. We do not want to type same loop each times.
# To remove that difficulty, we create a class with defined functions.
# We will use of classes, neral networks to automate the work.
# Hence, u should have at least basic knowledge of these libraries to understand this code.
# For class, any tutorial on any platform like youtube is enough. 
class PINN:
    def __init__(self,layers,lr=1e-3): # layers is mandatory entry, lr is 1e-3 by default and can be ajusted when calling the class. 
        # to build network automatically from list of layer widths
        # e.g.layers = [1,20,20,1] ::: 1 to 20 to 20 to 1. two input and output layers along with two hidden layers.
        net = [] # We follow this structre before introducing model specially if we have too many layers.
        # This structre is not mandatary if layers' number is fewer than 5 but by just changing the list layers, this:
        # reproduces another model with an optimizer and these automations are helpful when u need to be more professional.
        # u can define model and optimizer manually inside nn.Sequential if u want simple structre. 
        for i in range(len(layers)-1): # 4-1 = 3 steps transformation
            net.append(nn.Linear(layers[i],layers[i+1])) #1,20; 20,20; 20,1
            if i < len(layers)-2 : # last layer don't need nonlinearity
                net.append(nn.Tanh())
            self.model = nn.Sequential(*net)
            self.optimizer = torch.optim.Adam (self.model.parameters(),lr=lr)

    def forward(self,x):
        return self.model(x)
    def loss_pde (self,x):
        # u" + 1 = 0
        u = self.forward(x)
        u_x = torch.autograd.grad(u.sum(),x,create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(),x,create_graph=True)[0]
        return (u_xx+1).pow(2).mean()
    
    def loss_bc (self,x_b,u_b):
        return (self.forward(x_b)-u_b).pow(2).mean()
    def train_step (self,x,x_b,u_b):
        loss = self.loss_pde(x) + self.loss_bc(x_b,u_b)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
# We have successfully created a class that has every characteristic a neural network shoul have.
# We can call loss, training or whatever we want by just giving BCs, layer size,step etc.
# Usage to showcase how to use class PINN
# define neural network by layer structe

#e.g. 1.1: layers=[1,20,20,1] , Boundary conditions(BCs): u(0)=u(1)=0
"""pinn = PINN(layers=[1,20,20,1])
# Defining a neural network by the class may seem long at first.
# But if u observe carefully, class PINN has general structre that can be used for any neural network with any layer structre and BCs. 
# You can recreate class PINN for other pde by just modifying loss_pde while keeping everything else same.
# collocation points
x = torch.linspace(0,1,100).reshape(-1,1).requires_grad_(True)
# boundary conditions
x_b = torch.tensor([[0.],[1.]]) # x belongs to [0,1]
u_b = torch.tensor([[0.],[0.]]) # u(0)=u(1)=0

# training the model 
for step in range (3000):
    loss = pinn.train_step(x, x_b, u_b)
    # sample losses for validation
    if step % 500 == 0:
        print (f"step {step}: {loss}")
with torch.no_grad():
    print(pinn.forward(torch.tensor([[0.5]])).item()) # 0.125"""

#e.g. 1.2: layers=[1,50,50,50,1] with same BC as 1.1, let's check which converge faster
"""pinn = PINN(layers=[1,50,50,50,1])

# collocation points
x = torch.linspace(0,1,100).reshape(-1,1).requires_grad_(True)
# boundary conditions
x_b = torch.tensor([[0.],[1.]]) # x belongs to [0,1]
u_b = torch.tensor([[0.],[0.]]) # u(0)=u(1)=0

# training the model 
for step in range (3000):
    loss = pinn.train_step(x, x_b, u_b)
    # sample losses for validation
    if step % 500 == 0:
        print (f"step {step}: {loss}")
with torch.no_grad():
    print(pinn.forward(torch.tensor([[0.5]])).item())""" # 0.125
# It should be obvious that larger hidden layer converges faster.

# Let's change lr to see which lr converges faster without overshooting
pinn = PINN(layers=[1,50,50,50,1],lr=10) 

# collocation points
x = torch.linspace(0,1,100).reshape(-1,1).requires_grad_(True)
# boundary conditions
x_b = torch.tensor([[0.],[1.]]) # x belongs to [0,1]
u_b = torch.tensor([[0.],[0.]]) # u(0)=u(1)=0

# training the model 
for step in range (3000):
    loss = pinn.train_step(x, x_b, u_b)
    # sample losses for validation
    if step % 500 == 0:
        print (f"step {step}: {loss}")
with torch.no_grad():
    print(pinn.forward(torch.tensor([[0.5]])).item()) 
# The error never converge to a certian value if u use large lr.
# Rather, it starts to diverge after certain point.
# The optimum lr from hit and trial was closer to 1e-4 to 1e-2.
# Bonus level: Advanced tensor concepts / syntaxes

# syntax 1: torch.cat():
import torch

"""# e.g. 1
a = torch.tensor ([[1.],[2.],[3.]]) # shape (3,1)
b = torch.tensor ([[4.],[5.],[6.]]) # shape (3,1)

# case I: dim = 0, stack rows i.e. vertical join
c = torch.cat ([a,b],dim=0)
print ( f" c has shape: {c.shape}.\nc is: {c} ")
# case II: dim = 1, stack columns i.e. horizintol join
d = torch.cat ([a,b],dim=1)
print ( f" d has shape: {d.shape}.\nd is: {d} ")

# case III: negative indexing i.e. dim=-1 or dim=-2
# dim = -1 is equivalent to dim = 1 in 2D tensor
# dim = -2 is equivalent to dim = 0 in 2D tensor
x = torch.cat ([a,b],dim=-2)
print ( f" x has shape: {x.shape}.\nx is: {x} ")

# case IV: dim>1 or dim<-2, no such dimensions as 2D tensor has only 2 dimensions i.e. two total 4 indices including negative indentation
x = torch.cat ([a,b],dim=2)
print ( f" x has shape: {x.shape}.\nx is: {x} ")
# output: IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2"""

# syntax 2: torch.stack()
# e.g. 2
"""a = torch.tensor ([1.,2.,3.]) # shape (3,)
b = torch.tensor ([4.,5.,6.]) # shape (3,)

# case I: dim = 0, 
c = torch.stack ([a,b],dim=0)
print ( f" c has shape: {c.shape}; c is: {c} ")
# case II: dim = 1
d = torch.stack ([a,b],dim=1)
print ( f" d has shape: {d.shape}.;d is: {d} ")

# case III: negative indexing, only -1,-2 is allowed
x = torch.stack ([a,b],dim=-1)
print ( f" x has shape: {x.shape}.; x is: {x} ")

# case IV: dim < -2 or dim >1
y = torch.stack ([a,b],dim=2)
print ( f" x has shape: {x.shape}.; x is: {x} ")"""
# output: IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2

# e.g. 3: use of stack() and cat in combining boundary of 2D square or rectangle or any shape
n = 50
# collocation points
x_vals = torch.linspace (0,1,n)
y_vals = torch.linspace (0,1,n)
# 4 faces i.e. bottom, top, left, right
xy_bot = torch.stack ([x_vals, torch.zeros(n)],dim = 1)
xy_top = torch.stack ([x_vals,torch.ones(n)],dim=1)
xy_lft = torch.stack ([torch.zeros(n),y_vals],dim=1)
xy_rgt = torch.stack ([torch.ones(n),y_vals],dim=1)

# All four edges joined into tensor
xy_boundary = torch.cat ([xy_bot, xy_top, xy_lft, xy_rgt])
print(xy_boundary.size())
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
# part3: Time dependent equations

# Level 5: Heat equation: classic one dimensional second ordered problem in physics
# -u_t = alpha u_xx where d is partial differential operator

import torch
import torch.nn as nn
class Heat_PINN:
    def __init__(self, alpha=0.1, lr=1e-3):
        self.alpha = alpha
        self.model = nn.Sequential (
            nn.Linear(2,32),nn.Tanh(),
            nn.Linear(32,32),nn.Tanh(),
            nn.Linear(32,32),nn.Tanh(),
            nn.Linear(32,1)
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr
        )
    def forward(self,xt):
        # xt: shape (N,2), columns = [x, t]
        return self.model (xt)

    def loss_pde(self, xt):
        u = self.forward (xt)
        u_t = torch.autograd.grad (
            u.sum(), xt, create_graph=True)[0][:, 1:2]
        u_x =  torch.autograd.grad (
            u.sum(), xt, create_graph=True)[0][:, 0:1]
        u_xx =  torch.autograd.grad (
            u_x.sum(), xt, create_graph=True)[0][:, 0:1]
        return (u_t - self.alpha*u_xx).pow(2).mean()
    
    def loss_ic(self, x0):
        # initial condition: u(x,0) = sin (pi*x)
        import math
        t0 = torch.zeros(x0.shape[0], 1)
        xt0 = torch.cat([x0, t0], dim=1)
        u0 = torch.sin (math.pi * x0)
        return (self.forward(xt0)-u0).pow(2).mean()
    
    def loss_bc (self, t_pts):
        # Boundary: u(0,t) = u(1,t) = 0
        x0 = torch.zeros (t_pts.shape[0], 1)
        x1 = torch.ones (t_pts.shape[0], 1)
        xt0 = torch.cat ([x0, t_pts], dim=1)
        xt1 = torch.cat ([x1, t_pts], dim=1)
        return (self.forward(xt0).pow(2).mean() +
                self.forward(xt1).pow(2).mean())
    def train_step(self,xt,x0,t_pts):
        loss = (self.loss_pde(xt) + 
                10*self.loss_ic(x0) +
                10*self.loss_bc(t_pts))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# collocation by random
xt = torch.rand(200,2).requires_grad_(True)
x0 = torch.linspace(0,1,100).reshape(-1,1)
t_pts = torch.linspace(0,1,100).reshape(-1,1)

pinn = Heat_PINN(alpha=0.1)
# Check loss 
for step in range(5000):
    loss = pinn.train_step(xt,x0,t_pts)
    if step % 1000 == 0:
        print (f"step {step}: {loss}")

# Analytical: u(x,t) = exp(-alpha*pi2*t)*sin(pi*x)
# test at (0.5,0.5):
import math
exact = math.exp (-0.1*math.pi**2*0.5)*math.sin(math.pi*0.5)
with torch.no_grad():
    pred = pinn.forward(torch.tensor([0.5,0.5])).item()
    print(f"Exact: {exact}\n PINN: {pred}")
  # Level 6: setup initial conditions(IC) as hard constraint
# In level 5, we regulated violation of IC by giving more weight to loss due to IC.
# We can enforce IC by building it in output architecture also.
# If u(x,0) = g(x) then we define output network as u(x,t) = g(x) + t.N(x,t)
# This form reduces loss_IC in our network architecture

import torch
import torch.nn as nn
import math
class HardICPINN:
    def __init__(self, alpha=0.1, lr=1e-3):
        self.alpha = alpha
        self.model = nn.Sequential (
            nn.Linear(2,32),nn.Tanh(),
            nn.Linear(32,32),nn.Tanh(),
            nn.Linear(32,32),nn.Tanh(),
            nn.Linear(32,1)
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr)
    
    def forward(self, xt):
        x = xt[:,0:1]
        t = xt[:,1:2]
        raw = self.model(xt)
        # u(x,0) = g(x) = sin (pi*x)
        g = torch.sin (math.pi*x)
        return g + t*raw
    
    def loss_pde(self,xt):
        u = self.forward(xt)
        u_t = torch.autograd.grad(
            u.sum(),xt,create_graph=True)[0][:,1:2]
        u_x = torch.autograd.grad(
            u.sum(),xt,create_graph=True)[0][:,0:1]
        u_xx = torch.autograd.grad(
            u_x.sum(),xt,create_graph=True)[0][:,0:1]
        return (u_t - self.alpha*u_xx).pow(2).mean()
    
    def loss_bc (self, t_pts):
        # Boundary: u(0,t) = u(1,t) = 0
        x0 = torch.zeros (t_pts.shape[0], 1)
        x1 = torch.ones (t_pts.shape[0], 1)
        xt0 = torch.cat ([x0, t_pts], dim=1)
        xt1 = torch.cat ([x1, t_pts], dim=1)
        return (self.forward(xt0).pow(2).mean() +
                self.forward(xt1).pow(2).mean())
    
    def train_step(self, xt, t_pts):
        loss = (self.loss_pde(xt) + 10*self.loss_bc(t_pts))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

xt = torch.rand (2000,2).requires_grad_(True)
t_pts = torch.linspace (0,1,100).reshape (-1,1)

pinn = HardICPINN (alpha=0.1)
for step in range (5000):
    loss = pinn.train_step(xt,t_pts)
    if step % 1000 == 0:
        print (f"step {step}: {loss}")

# test at (0.5,0.5):
import math
exact = math.exp (-0.1*math.pi**2*0.5)*math.sin(math.pi*0.5)
with torch.no_grad():
    pred = pinn.forward(torch.tensor([[0.5,0.5]])).item()
    print(f"Exact: {exact}\n PINN: {pred}")
# if u remember we did pred = pinn.forward(torch.tensor([0.5,0.5])).item() in level 5 i.e. shape (2,)
# but pred = pinn.forward(torch.tensor([[0.5,0.5]])).item() here has shape (2,1)
# These concepts come under the error handling part and are outside the scope of tutorial as:
# our aim here is to build functional understanding only. If I feel like uploading more crucial parts, I may mention it in future.
# section 4: Inverse problem (heart of PINN)

# Level 7: recovering an unknown parameter
# in forward problem pde, parameters is known and we aim to train our model to determine the solution.
# for example: u(x,t) has pde u_t = alpha u_xx in case of heat equation where alpha is known.

# In reverse problem portion u(x,t) is known usually through experiments.
# So, our aim is to recover alpha from given piece of information.
# For solving pde there are various numerical methods we may not pde.
# However, recovering information some kind of scrammbled dataset is tricky choice.
# So, PINN shines more at reverse problems involving noise, incomplete data etc.

import torch
import torch.nn as nn
import math

class InversePINN:
    def __init__(self, alpha_init = 0.5,lr=1e-3):
        # alpha_init is initial guess and arbitary to programmers choice
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.net = nn.Sequential (
            nn.Linear(2,32), nn.Tanh(),
            nn.Linear(32,32), nn.Tanh(),
            nn.Linear(32,1))
        self.optimizer = torch.optim.Adam (
            list(self.net.parameters()) + [self.alpha], lr=lr)
        
    def forward (self, xt):
        return self.net(xt)
    
    def loss_pde (self,xt):
        u = self.forward(xt)
        u_t = torch.autograd.grad(
            u.sum(),xt,create_graph=True)[0][:,1:2]
        u_x = torch.autograd.grad(
            u.sum(),xt,create_graph=True)[0][:,0:1]
        u_xx = torch.autograd.grad(
            u_x.sum(),xt,create_graph=True)[0][:,0:1]
        return (u_t - self.alpha*u_xx).pow(2).mean()
    
    def loss_data (self, xt_data, u_data):
        return (self.forward(xt_data)-u_data).pow(2).mean()
    
    def train_step(self, xt, xt_data, u_data):
        loss = self.loss_pde(xt) +10*self.loss_data(xt_data,u_data)
        self.optimizer.zero_grad()
        loss.backward ()
        self.optimizer.step()
        return loss.item(), self.alpha.item()
    
# Generate data from true solution with alpha = 0.1
true_alpha = 0.1
n_data = 50
xt_data = torch.rand(n_data, 2)
x_d, t_d = xt_data[:,0], xt_data[:,1]
u_clean = torch.exp(-true_alpha*math.pi**2*t_d) * torch.sin(math.pi*x_d)
noise = 0.01*torch.rand(n_data)
u_data = (u_clean+noise).reshape(-1,1)

# collocation points 
xt = torch.rand(2000,2).requires_grad_(True)

pinn = InversePINN(alpha_init=0.5) # initial guess which is generally wrong
for step in range(6000):
    loss, alpha_est = pinn.train_step (xt, xt_data, u_data)
    if step % 1000 == 0:
        print (f"step {step}: loss = {loss}")

print(f"True_alpha = {true_alpha} | recovered alpha = {alpha_est}")
# In next level we will discuss when coefficients are themselves are function of time.
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

# Thank u for your patience. The goal of this tutorial is served.
# If I make useful content in future, I will let you know.
