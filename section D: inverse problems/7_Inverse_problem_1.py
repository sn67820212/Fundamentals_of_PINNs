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