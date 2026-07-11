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
