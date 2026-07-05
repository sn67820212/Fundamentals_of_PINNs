import torch
import torch.nn as nn

# This part will show you a professional structre of PINN.
# The reader is expected to know few basic syntaxes of torch and class.

# Section A: Architecture (level 1 and level 2)

# Level 2: weighted loss terms.
# When you add multiple loss terms, they have different magnitudes.
# For example: boundary loss of 1e-4 and pde loss of 1e0 means optimizer ignores boundary.
# The ignorance of certain losses can give models which are not physical at all.
# To remove this problem, weighted loss is used.
# In simple words, weighted loss terms are used to give desired degree of prority to loss terms.
# Another instance when we could use weighted loss is if certain source from which loss is computed is incomplete.
# For example, we have few BC data but complete pde thus collocation points are used uniformly.
# If we have only half of the BC known then giving double weight to loss_bc can be an option.

# e.g. : 
class PINN:
    def __init__(self,layers,lr=1e-3,w_pde=1.0,w_bc=10): # forces bc harder as it's given more weight
        net = []
        for i in range(len(layers)-1): 
            net.append(nn.Linear(layers[i],layers[i+1]))
            if i < len(layers)-2 :
                net.append(nn.Tanh())
        self.model = nn.Sequential(*net)
        self.optimizer = torch.optim.Adam (self.model.parameters(),lr=lr)
        self.w_pde = w_pde
        self.w_bc = w_bc
    def forward (self,x):
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
        L_pde = self.loss_pde(x)
        L_bc = self.loss_bc(x_b,u_b)
        loss = L_pde*self.w_pde + L_bc*self.w_bc
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # return individual losses for dignosis
        return L_pde.item(), L_bc.item()
    # note: when we pass different objects, objects with different size the error/bug may occur
    # For example train_step() returns two values which means loss terms are obtained as list.
    # Getting list is not bad but not knowing and throwing operations that aren't compatible with list causes error.
    # Also, even if u get output but there could be bug as u did non list logical reasoning upon list.

# e.g. use class PINN to model pde with BCs, u(0)=0, u(1) = 0
pinn = PINN(layers=[1,20,20,1])

# collocation points
x = torch.linspace(0,1,100).reshape(-1,1).requires_grad_(True)
# boundary conditions
x_b = torch.tensor([[0.],[1.]]) # x belongs to [0,1]
u_b = torch.tensor([[0.],[0.]]) # u(0)=u(1)=0

# training the model 
for step in range (3000):
    l_pde, l_bc = pinn.train_step(x, x_b, u_b) # converted list output to scaler form
    # sample losses for validation
    if step % 500 == 0:
        print (f"step {step}: Loss_pde={l_pde}; Loss_BC = {l_bc}")
with torch.no_grad():
    print(pinn.forward(torch.tensor([[0.5]])).item()) 
# changing the value of weights changes rate of convergence  
# If the loss isn't getting reduced for certain types of losses, it's bad news.
# From now on, u may try different weights to capture good model from training. 