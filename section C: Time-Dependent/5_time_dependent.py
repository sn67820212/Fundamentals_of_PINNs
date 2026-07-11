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