# %%importing required libraries
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%%%%%%%%%%%%%%%%%%%%%%%% TENSOR INITIALISATIONS %%%%%%%%%%%%%%%%%%%%%%%%%

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device)

# other ways for initialising tensors
tensor = torch.empty(size=(3, 3))
random_tensor = torch.rand(size=(2, 2))
zeros_tensor = torch.zeros(size=(3, 3))
identity_tensor = torch.eye((2))
arange_tensor = torch.arange(start=0, end=4, step=1)
linspace_tensor = torch.linspace(start=0.1, end=1, steps=10)
normaldist_tensor = torch.empty(size=(1, 6)).normal_(mean=0, std=1)
uniformdist_tensor = torch.empty(size=(1, 6)).uniform_(0)
diagonal_tensor = torch.diag(torch.ones(5))

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.shape)

# converting tensors to other data types after initialising
my_tensor.bool()
my_tensor.short()
my_tensor.long()
my_tensor.half()
my_tensor.float()
my_tensor.double()

# converting numpy arrays to tensors and vice-versa

my_array = np.zeros((5, 5))
my_tensor = torch.from_numpy(my_array)
my_new_array = my_tensor.numpy()

# %%%%%%%%%%%%%%%%%%%%%%%%% TENSOR MATHEMATICS METHODS %%%%%%%%%%%%%%%%%%%%%%%%%

t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([4, 5, 6])

# -----------------Addition and Subtraction-----------------

# 1st method:
t = torch.empty(3)
t = torch.add(t1, t2)

# 2nd method:
t = t1 + t2

# -----------------Division-----------------
t = torch.true_divide(t1, t2)

# Inplace Operations (_ is followed at the end of the method)
t1.add_(t2)
t1.subtract_(t2)

# -----------------Exponentiation-----------------

# 1st method
t = t1.pow(2)

# 2nd method
t = t1 ** 2

# -----------------Matrix Multiplication-----------------
t1 = torch.rand((2, 5))
t2 = torch.rand((5, 3))

# 1st method
t3 = torch.mm(t1, t2)

# 2nd method
t3 = t1.mm(t2)

# -----------------Matrix Exponentiation-----------------
t1 = torch.rand((3, 3))

t2 = t1.matrix_power(2)

# -----------------Element Multiplication-----------------
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([4, 5, 6])

t = t1 * t2

# -----------------Dot Product-----------------
t = torch.dot(t1, t2)
