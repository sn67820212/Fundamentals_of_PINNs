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