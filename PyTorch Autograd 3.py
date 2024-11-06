# بسم الله الرحمن الرحيم
"""
How to take derivatives/gradients w.r.t NON-leaf tensors (intermediate weights/parameters)

Eg:
    x = 2  --> leaf
    y = 2 * x  --> non-leaf
    z = y**2

    * ∂z/∂x = 8 * x  --> gradient w.r.t to a leaf node
    * ∂z/∂y = 2 * y  --> gradient w.r.t to a non-leaf node
      ∂z/∂y = 2 * (2 * x)
      ∂z/∂y = 4 * x
"""
import torch

# "x" here is a leaf node (leaf nodes/weights always require grad by default)
x = torch.tensor([1, 2, 3.], requires_grad=True)  # *** Note: "torch.Tensor()" class constructor doesn't take the flag "requires_grad" ***
print(f"x is leaf?: {x.is_leaf} \n")

y = 2 * x  # ANY computation on any leaf node results in a non_leaf output without a ".grad" attribute (unless "retain_graph" is specified)
print(f"y is leaf?: {y.is_leaf}")
print(f"y requires gradient?: {y.requires_grad} \n")

z = y**2
print(f"z is leaf?: {z.is_leaf}")
print(f"z requires gradient?: {z.requires_grad} \n")

y.retain_grad()  # --> Has to be called before calling ".backward()" and after applying all the computations on "y"

z.backward(torch.ones(3))  # Had to pass a (1x3) tensor to match shape of the output "z"
print(f"∂z/∂x = {x.grad} \n∂z/∂y = {y.grad}")
