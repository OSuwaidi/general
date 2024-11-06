# بسم الله الرحمن الرحيم
# PyTorch practice

import torch
import numpy as np
import time

t_p = time.perf_counter()

# What is the difference between "torch.tensor" and "torch.Tensor"?
# --> "torch.Tensor()" defaults the data type of the output as float32 (32 bits), where on the other hand; "torch.tensor()" takes the data type of its inputs
# (Recall: bit is a binary piece of storage, that stores data as either 0 or 1)
# Example:
v = torch.Tensor([1, 2, 3])
w = torch.tensor([1, 2, 3])
print(f"v dtype = {v.dtype} \nw dtype = {w.dtype} \n")  # Notice that even though the inputs are integers, torch.Tensor by default sets them as floats of 32 bits of storage
                                                     # Whereas torch.tensor kept the data type as int64 (64 bits of data)

# This is VERY important as you cannot multiply matrices of different data types
# Example:
"""
ans = v.t() @ w  --> This would give an error, because we are multiplying data type float32 with int64 (not compatible)
ans = w @ v.t() --> Same as above
"""
r = torch.rand(3)  # torch.rand(3) gives a (3x1) array of type float32
ans = v.t() @ r
print(f"v_t * r = {ans} \n\n")
"""
However, w.t() @ r  --> This would give an error, because we are multiplying data type int64 with float32 (not compatible)
"""


# Creating a diagonal matrix
eigen = torch.tensor([3] * 3)
print(f"eigen value = {eigen}")
diag = torch.diag(eigen)
print(diag, "\n\n")


# Tensor/matrix indexing works the same way as list indexing, you start from [0, 0] not [1, 1]  --> [r, μ]
A = torch.Tensor([[1]*3,
                  [2]*3,
                  [3]*3])
print(f"A = {A}")

A[1, 1] = 5
print(f"A_new = {A} \n")  # Notice that it went to the middle index, not top left

# To take a row vector out of a matrix:
print(f"v1 = {A[0]} \nv2 = {A[1]} \nv3 = {A[2]} \n")

# To take a column vector out of a matrix:
print(f"v1 = {A[:, 0]} \nv2 = {A[:, 1]} \nv3 = {A[:, 2]} \n")

# However, the element taken from matrix indexing is not a python scalar (int/float)! It is still of type array/Tensor:
print(f"A[1, 1] = {A[1, 1]} of type: {type(A[1, 1])}")
# To make it a python scalar:
print(f"A[1, 1] = {A[1, 1]} of type:{type(A[1, 1].item())} \n\n")  # ".item()" gives back python scalar for a single index ONLY!
# We could also use ".tolist()" if the tensor contains more than a single value


# Using "torch.arange()" (array range), which is basically the "range" function but for arrays:
arr = np.arange(10, dtype=np.float32).reshape(5, 2)  # Make it 5 rows and 2 columns
print(f"arr = {arr}")

# Note: the number of elements in your matrix/tensor should match the multiplication of the inputs in the ".view(r, μ)" method --> (# = r*μ)
t = torch.arange(10, dtype=torch.float32).view(5, 2)  # Make it 5 rows and 2 columns  --> ".view()" is equivalent to ".reshape()" in numpy
print(f"t = {t} \n")


# Note that by default, numpy arrays are set to data type int64
n = np.arange(10)
print(f"n dtype = {n.dtype}")

t1 = torch.tensor(n)  # Keeps data type as is
t2 = torch.Tensor(n)  # Will convert data type to float32
print(f"t1 dtype = {t1.dtype}")  # "torch.tensor()" preserves the data type of its inputs, thus it remains int64
print(f"t2 dtype = {t2.dtype} \n")  # But "torch.Tensor()" will convert the data type into float32
print(f"Time = {time.perf_counter() - t_p:.4} seconds")
