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
pinn = PINN(layers=[1,50,50,50,1])

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
    print(pinn.forward(torch.tensor([[0.5]])).item()) # 0.125
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
