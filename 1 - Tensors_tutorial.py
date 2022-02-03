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

# ----------------- Batch Matrix Multiplication-----------------
t = torch.dot(t1, t2)

# ----------------- Batch Matrix Multiplication-----------------
batch = 3
n = 1
m = 2
p = 3
t1 = torch.rand((batch, n, m))
t2 = torch.rand((batch, m, p))
bmm = torch.bmm(t1, t2)

# ----------------- Broadcasting -----------------
# in the example below, t2 is being subtracted from all 5 rows in t1
t1 = torch.rand((5, 5))
t2 = torch.rand((1, 5))

t = t1 - t2
t = t1 ** t2

# ----------------- Sum Method -----------------
# Get a scalar
torch.sum(t1)

# To sum over all rows (i.e. for each column)
torch.sum(t1, dim=0)

# To sum over all columns (i.e. for each row)
torch.sum(t1, dim=1)

# ----------------- Max & Min  -----------------

# returns the values and their indices
torch.max(t1, dim=1)
torch.min(t1, dim=0)

# returns the indices of the max/min value of all elements in the input
torch.argmax(t1)
torch.argmin(t1, dim=0)

# ----------------- Other Useful Maths Methods -----------------
# converts the tensor elements to absolute values
torch.abs(t1)

# returns the mean
torch.mean(t1.float(), dim=0)

# returns true if any two elements are equal, otherwise false
torch.eq(t1, t2)

# sort the tensor values in the specified order
torch.sort(t1, descending=False)

# it will change the values of any elements that are below the min and above the max specified
torch.clamp(t1, min=0, max=10)

# we can check the values of a bool type tensor
t4 = torch.tensor([0, 1, 1, 1, 0], dtype=torch.bool)
torch.any(t4)
torch.all(t4)

# %%%%%%%%%%%%%%%%%%%%%%%%% TENSOR INDEXING %%%%%%%%%%%%%%%%%%%%%%%%%

batch_size = 10
features = 25
t5 = torch.rand((batch_size, features))

# all features of the first example in the batch
t = t5[0]  # equivalent to t5[0,:]
t = t5[0].shape

# the first feature of all batches
t = t5[:, 0].shape

# the first 10 features of the third example of the batch
t = t5[2, : 10].shape

# return the elements with an indix that match the elements in the indices list
t = torch.arange(10)
ind = [2, 5, 8]
print(t[ind])

# returns the two elements with indices (1,3) and (0,1) in the given tensor
t = torch.rand((5, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([3, 1])
print(t[rows, cols])

# returns the elements in the tensor that are less than 2 and more than 8
t = torch.arange(10)
print(t[(t < 2) | (t > 8)])

# returns the elements in the tensor that have remainder 0 when divided by 2
t = torch.arange(10)
print(t[t.remainder(2) == 0])

# if the elements dont meet the condition then an alternative value is gives as per 3rd attribute
# if they do meet the condition, they are returned as they are
print(torch.where(t > 5, t, t * 2))

# returns the unique elements in a tensor
t.unique()

# returns the dimension of a tensor
t.ndimension()

# count the number of elements in a tensor
t.numel()

# %%%%%%%%%%%%%%%%%%%%%%%%% TENSOR RESHAPING %%%%%%%%%%%%%%%%%%%%%%%%%

# reshaping the tensor into a matrix of 3x3
t = torch.arange(9)
# 1st method
t1 = t.view(3, 3)

# 2nd method
t2 = t.reshape(3, 3)

# transpose of the tensor matrix
t3 = t2.t()

# concatenate tensors
t1 = torch.rand((2, 3))
t2 = torch.rand((2, 3))

t = torch.cat((t1, t2), dim=0)

# flatten the matrix completely into one dimension
t4 = t.view(-1)

# squeeze and unsqueeze?
