import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%initialising tensor
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.shape)

# other ways for initialising tensors
my_tensor = torch.empty(size=(3, 3))
my_tensor = torch.rand(size=(2,2))
my_tensor = torch.zeros(size=(3, 3))
my_tensor = torch.eye((2))
my_tensor = torch.arange(start=0, end=4, step=1)
my_tensor = torch.linspace(start=0.1, end=1, steps=10)
my_tensor = torch.empty(size=(1, 6)).normal_(mean=0, std=1)
my_tensor = torch.empty(size=(1, 6)).normal_(mean=0, std=1)
my_tensor = torch.diag(torch.ones(5))

print(my_tensor)

# %%converting tensors to other data types after initialising
my_tensor = torch.arange(5)

my_tensor.bool()
my_tensor.short()
my_tensor.long()
my_tensor.half()
my_tensor.float()
my_tensor.double()


# %%converting numpy arrays to tensors and vice-versa

my_array = np.zeros((5,5))
my_tensor = torch.from_numpy(my_array)
my_array2 = my_tensor.numpy()

# %%