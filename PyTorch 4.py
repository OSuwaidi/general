# بسم الله الرحمن الرحيم

from __future__ import print_function
import torch
import numpy as np

# Construct a randomly initialized matrix:
x = torch.rand(5, 3)
print(x, "\n")

# Construct a matrix filled with zeros and of dtype=long:
x = torch.zeros(5, 3, dtype=torch.long)
print(x, "\n")

# Construct a tensor directly from data:
x = torch.tensor([5.5, 3, 1])
print(x, "\n")

# Create a tensor based on an existing tensor. These methods will reuse some of the properties of the input tensor: "dtype", "device"; unless new values are provided by user
x = x.new_ones(5, 3, dtype=torch.double)  # "torch.double" == float64 (64 bits of data)
print(x, "\n")

x = torch.randn_like(x, dtype=torch.float)  # Override dtype! (float alone implies float32)
print(x)  # Result has the same size, but with different values because we provided "torch.randn_like"
print(x.dtype, "\n")  # And different dtype

# To get a tensor's size:
print(x.size())
r, c = x.size()  # Tuple unpacking!
print(r, c, "\n")


#########################################################################################################
# Operations:

# To copy a tensor as it is:
x = torch.tensor([1, 2, 3])
print(x)
y = x.clone()
print(y, "\n\n")

# Normal addition:
z = x + y
print(z)

# Addition: providing an output tensor as argument:
result = torch.empty(3)  # If x, y were (mxn) for example --> torch.empty(m, n)
torch.add(x, y, out=result)  # This makes it faster to store data than creating it form scratch
print(result)

# Addition: in-place: Any operation that mutates a tensor in-place is post-fixed with a "_()". For example: x.copy_(y), x.t_(): will change original "x"!!! --> form: "tensor.operation_()"
y.add_(x)  # Adds "x" onto "y"
print(y, "\n")
y.resize_(3, 1)
print(y, '\n')

# Using indexing (recall indexing starts from 0):
x = torch.randint(0, 11, (3, 3))
print(x, "\n")

# Grab the second column:
print(x[:, 1])

# Grab the second row:
print(x[1], "\n")

# To resize/reshape a tensor:
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(2, -1)  # "-1" implies that # the size is inferred from other dimensions
print(x.size(), y.size(), z.size(), "\n")

# To grab a Python number/scalar from one element tensor:
x = torch.randn(1)
print(x)
print(x.item(), "\n\n")


#########################################################################################################
# Converting from numpy to torch and vice-versa:
# The Torch Tensor and NumPy array will share their underlying memory locations (if the Torch Tensor is on CPU), and changing one will change the other:
a = torch.ones(5)
print(f"a = {a}")

b = a.numpy()
print(f"b = {b} \n")

a[1] = 2  # Changing the torch Tensor
print(a)
print(b, "\n")  # Notice that ndarray "b" changed by only changing tensor "a"

a.add_(1)  # Recall this is an "in-place" addition with "_()", thus the original torch "a" is affected
print(a)
print(b, "\n\n")

# Going the other way around:
a = np.ones(5)
b = torch.from_numpy(a)
a += 2
print(a)
print(b)


#########################################################################################################
# CUDA Tensors: Tensors can be moved onto any device using the ".to()" method
# CUDA is a parallel computing platform and application programming interface model created by Nvidia. It allows software developers to use a CUDA-enabled graphics processing unit (GPU) for general purpose processing
# It utilizes the concept of parallel processing
"""
# let us run this cell only if CUDA (NVIDIA GPU) is available 
# We will use "torch.device" objects to move tensors in and out of GPU
if torch.cuda.is_available():  --> False for Macs
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
"""