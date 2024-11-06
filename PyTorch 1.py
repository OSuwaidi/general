# بسم الله الرحمن الرحيم
# "A" is a matrix that has a null space = 2, when multiplied by vector set V
# Implies: Ker(A) = 2, i.e: there are 2 sets of vectors (v and w) contained in V that go to the zero vector when transformation A is applied on them
# Note that all linear combinations of v and w (span of v, w) also point to the zero vector (i.e: vector "u" in our example)

import numpy as np
import torch
import time

start = time.perf_counter()

A = torch.Tensor([[1, 1, 1, 1],
                  [1, 2, 3, 4],
                  [4, 3, 2, 1]])
v = torch.Tensor([1, -2, 1, 0])  # *** Note ***: Surrounding the elements in "Tensor" with single brackets [ ]: makes it a 1D Tensor!!! --> (4,) is NOT (4, 1); whereas surrounding the elements in "Tensor" with double brackets [[ ]]: makes it a 2D ROW-vector (1x4)
w = torch.Tensor([2, -3, 0, 1])  # Thus, we can do: w = torch.Tensor([[2, -3, 0, 1]])  --> A @ w.T  (more precise)  --> single bracket = 1 axis, double brackets = 2 axes
u = 3 * v + -1.2 * w  # Q.) Doesn't a "linear combination of 2 independent vectors" (v and w) mean that they span all of R^2 space, and their linear comb. can produce ANY vector, thus [A * (any_vector)] should also be = 0?
# Ans: Yes they do span all of R^2 space, but that doesn't mean that all vectors will be sent to the zero vector, because matrix A has 4 columns, meaning it applies only on R^4 vectors (4 dimensional vectors only, not 2). And for your 2 independent vectors to produce any vector in R^4 (span R^4), they would need 2 more independent vectors to span all of R^4. i.e, to span all of R^4, you would need a linear combination of 4 linearly independent vectors, not only 2.
# Thus, if you want to produce vectors that span all of R^N space, then you would need a linear combination of at least N independent vectors!!!

# Random information: print(A.view(4, 3)) ---> Converts matrix A from (3x4) to (4x3), looking at the first row of matrix A, it will stop at the 3rd element, then shift the rest into the row below

print(f"A = {A} \nv = {v} \nw = {w} \nu = {u} \n\nA size = {A.size()} \nv size = {v.size()} \nw size = {w.size()} \nu size = {u.size()} \n")
print(f"A * v = {torch.matmul(A, v)} \nA * w = {torch.matmul(A, w)} \nA * u = {torch.matmul(A, u)} \n\n")  # An alternative to "torch.matmul()": "A @ v"

x = torch.rand(4, 1)  # Note: "x = torch.rand(4)" gives a horizontal vector (non-row vector)

print(f"x = {x} \n \nx size = {x.size()} \n\nA * x = {A @ x} \n\n")

######################################################################################################
# To Transpose a matrix:
B = torch.tensor([[5, 7, 1, 2, 3],
                  [10, 14, 2, 4, 6]])

print(f"B = {B} \n\nB size = {B.size()} \n")

print(f"B transposed = {B.transpose(0, 1)} \n\nB transposed size = {B.size()} \n")  # Notice that size B is still the original size (2 x 5), because torch.transpose doesn't affect the original matrix

# Or you can transpose using shorthand method:
print(f"B trans = {B.t()} \n")
print(f"B Trans = {B.T} \n")

# Note: A * B = (B^T * A^T)^T
A = torch.rand(3, 3)
B = torch.rand(3, 3)

print(A @ B)
print((B.t() @ A.t()).t(), "\n\n")

######################################################################################################
# To Inverse a matrix (matrix must be a square matrix with non-zero determinant):
C = torch.rand(3, 3) * 10  # You can only find the determinant of a Tensor if, and only if that Tensor was type float!!!
print(f"C = {C} \n\nC inverse = {C.inverse()} \n\n")  # Alternatively, you could use torch.inverse(C)

######################################################################################################
# To find the determinant of a matrix:
print(f"C determinant = {C.det()} \n\n")  # Alternatively, you could use torch.det(C)

######################################################################################################
# To convert a "numpy ndarray" to "torch Tensor":
a = np.array([1, 2, 3])
print(f"a = {a}")
print(type(a))
print(a.dtype, "\n")  # dtype = data type

t = torch.from_numpy(a)
print(f"t = {t}")
print(type(t))
print(t.dtype)  # IMPORTANT: They both share the SAME MEMORY, thus modifying any element of 'Tensor' will also be reflected in 'ndarray', and vice-versa

# Or another, but MUCH slower way:
t = torch.tensor(a)  # Or t = torch.Tensor(a) to make it float32
print(f"t_slower = {t} \n\n")

######################################################################################################
# To convert a "torch Tensor" to "numpy ndarray":
T = torch.tensor([[1] * 3,
                  [2] * 3,
                  [3] * 3])
print(f"t = {T} \nT_type = {type(T)}\n")

n = t.numpy()
print(f"n = {n} \nn_type ={type(n)} \n\n")

######################################################################################################
# To create a 3D tensor of size (2x3x2):
T = torch.tensor([[[1, 2],
                   [3, 4],
                   [5, 6]],  # First (front) layer

                  [[7, 8],
                   [9, 10],
                  [11, 12]]])  # Second (back) layer
print(f"T = {T} \nT shape = {T.shape}\n")

# OR:
T = torch.randint(0, 10, (3, 2, 5))  # Creates a (2x5) matrix with 3 layers (depth) --> "torch.randint(low, high, (dim, r, μ))
print(f"T = {T} \n\n")

######################################################################################################
# To change tensor's datatype:
x = torch.rand(3, 3)
print(x.dtype)
x = x.type(torch.int32)
print(x.dtype, '\n\n')

######################################################################################################
# Understanding "torch.rand()" size* parameter arguments:
x = torch.rand(2, 3, 2, 3)  # --> "torch.rand((# batches, # dimensions/layers/channels, height/row, width/column))
y = torch.rand(3, 2, 3, 3)  # --> The first element represents the number of "batches" (roughly), while the second represents the number of "layers/dimensions", and the last two represent height/rows and width/columns.
print(f"x = {x} \n\ny = {y} \n\n")
# *** Note ***: If matrices are separated by single space ==> Different depths/layer of same batch
# *** Note ***: If matrices are separated by double spaces ==> Different batches

######################################################################################################
# To create multiple single array tensors from a multi-array tensor:
a = torch.tensor([1, 2, 3])
x, y, z = a.unbind()  # Tuple unpacking!
print(f"x = {x} \ny = {y} \nz = {z} \n")

print(f"Time Performance = {time.perf_counter() - start:.4} seconds")
