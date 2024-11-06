# بسم الله الرحمن الرحيم

import torch
import numpy as np

I = torch.Tensor([1] * 3)
I = torch.diag(I)
X = torch.rand(1, 3, 3)  # ".rand(layers/depth = 1, r, μ)" --> rand()" gives random floats from 0 to 1 (only positive values). While "randn()" gives random floats from the standard normal distribution (mean=0, variance=1) (can be negative)
print(f"I = {I}")
print(f"X = {X} \n")
print(f"X * I = {X @ I}")
print(f"I * X = {I @ X} \n")

A = torch.randint(0, 10, (3, 3))  # "randint()" --> (low, high, (depth=1, r, μ)) --> gives random integers values from 0 to 9 (excluding 10) of size (3x3)
print(f"A_int = {A} \n")
'''
A_inv = A.inverse()
print(A @ A_inv, "\n", A_inv @ A) --> Both of these result in errors because "A.inverse()" EXPECTS the inverse matrix to be also of type integer, like its original matrix A, but it's not, it should be of type float!
'''
# To fix this issue:
A = torch.Tensor(np.random.randint(0, 10, (3, 3)))  # Recall: "torch.Tensor()" returns type float32 tensor/matrix, while "torch.tensor()" is of type int64 by default
print(f"A_flt = {A} \n")
A_inv = A.inverse()  # Recall: a "singular" matrix (zero determinant) has NO inverse
print(f"A * A_inv = {A @ A_inv}")

# OR:
A = torch.randint(0, 10, (3, 3), dtype=torch.float32)  # We force the data type of matrix A to be of type float32
A_inv = A.inverse()
print(f"A * A_inv = {A @ A_inv} \n")

# Python float system:
print(A @ A_inv == I, "\n\n")  # We get "False" even though A * A_inv = I, because python can't add floats perfectly, there is always some rounding up error!


#########################################################################################################################
# Statistical information:

x = [1, 2, 1.5, 3, 4, 2.1, 0.9]
x_bar = sum(x)/len(x)
sx = 0
for i in x:
    sx += (i - x_bar)**2
sx = (sx / len(x))**0.5
print(sx)

y = [5, 3, 2.5, 7, 8, 6.5, 4]
y_bar = sum(y)/len(y)
sy = 0
for i in y:
    sy += (i - y_bar)**2
sy = (sy / len(y))**0.5
print(sy)

cov = 0
for xi, yi in zip(x, y):
    cov += (xi - x_bar)*(yi - y_bar)
cov = cov / len(x)
print(cov)

cor = cov / (sx*sy)
print(cor)
