# بسم الله الرحمن الرحيم
# Exploring the AUTOGRAD: AUTOMATIC DIFFERENTIATION package

import torch


a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)  # False, because "requires_grad" is set to False by default. ".requires_grad_()" changes an existing Tensor's "requires_grad" flag to True (in-place operation).
a.requires_grad_()  # --> sets tensor a's "requires_grad=True"
print(a.requires_grad)

b = (a * a).sum()  # "tensor.sum()" adds all the elements in a tensor together
print(b.grad_fn, "\n\n")


'''
The autograd package provides automatic differentiation for all operations on Tensors. 
It is a define-by-run framework, which means that your backprop is defined by how your code is run, and that every single iteration can be different.
"torch.Tensor" is the central class of the package. If you set its attribute ".requires_grad" as True, it starts to track all operations on it.
When you finish your computation you can call ".backward()" and have all the gradients computed automatically.
The gradient for this tensor will be accumulated into the leaf ".grad" attribute.

To stop a tensor from tracking history, you can call ".detach()" to detach it from the computation history, and to prevent future computation from being tracked.

To prevent tracking history (and using memory), you can also wrap the code block in "with torch.no_grad():"

If you want to compute the derivatives, you can call ".backward()" on a Tensor.
If Tensor is a scalar (i.e. it holds a one element data), you don’t need to specify any arguments into ".backward()", however if it has more elements, you need to specify a "gradient" argument, that is a tensor of matching shape.
'''


x = torch.ones(2, 2, requires_grad=True)
print(f"x = {x}")
y = x + 2  # Adds 2 to each element in "x"
print(f"y = {y} \n")

# Since "y" was created as a result of an operation (addition), it has a "grad_fn" (memory location), and we can track it back:
print(y.grad_fn, "\n")

z = 3 * y**2  # Apply some operations on "y"
out = z.mean()  # Finds the mean (average) of all the elements in the tensor
print(f"z = {z}")
print(f"out = {out} \n")

# Let’s backprop now. Because "out" contains a single scalar, thus "out.backward()" is equivalent to "out.backward(torch.tensor(1.))"
out.backward()  # ***IMPORTANT***: "grad" can be implicitly created ONLY for scalar outputs (single index)

print(f"∂out/∂x = {x.grad} \n\n")  # Prints gradients w.r.t to variable vector "x" --> ∂(out)/∂x --> (NOT only for a specific variable "x_i" in variable vector "x")
# print(y.grad)  -->  The ".grad" attribute of a non-leaf Tensor DOES NOT EXIST!!!  To access the gradient of non-leaf use ".retain_grad()"
'''
* Gradient enabled tensors (variables) along with functions (operations) combine to create the dynamic computational graph. 
* "grad": "grad" holds the value of the gradient. If "requires_grad" is False it will hold a "None" value. Even if "requires_grad" is True, it will hold a "None" value UNLESS ".backward()" function is called from some other node. 
* For example, if you call "out.backward()" for some variable "out" that involved "x" in its calculations, then "x.grad" will hold ∂(out)/∂x.

The "out" tensor was formed by multiple operations:
out = 1/4 * (SUM_i (z_i))
out = 1/4 * (SUM_i (3 * y_i**2))
out = 1/4 * (SUM_i (3 * (x_i + 2)**2))

Remove the "SUM" to find the derivative w.r.t each individual element (x_i):
out = 1/4 * (3 * (x_i + 2)**2)

∂(out)/∂x_i = 2/4 * (3 * (x_i + 2))
∂(out)/∂x_i = 6/4 * (x_i + 2)
∂(out)/∂x_i = 3/2 * (x_i + 2)  --> x_i = 1
∂(out)/∂x_i = 3/2 * (1 + 2)
∂(out)/∂x_i = 9/2
∂(out)/∂x_i = 4.5

*********************************************************************************

Also could've found it using the Chain Rule:
out = 1/4 * (SUM_i (z_i))
z_i = 3 * y_i**2
y_i = x_i + 2

We want to find ∂(out)/∂x_i:
∂(out)/∂x_i = [∂(out)/∂z_i] * [∂z_i/∂y_i] * [∂y_i/∂x_i]

∂(out)/∂z_i = 1/4
∂z_i/∂y_i = 6 * y_i
∂y_i/∂x_i = 1 + 0

∂(out)/∂x_i = (1/4) * (6 * y_i) * (1)  --> y_i = 3
∂(out)/∂x_i = (1/4) * (18) 
∂(out)/∂x_i = 92 = 4.5


*********************************************************************************


https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py
https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95

* Mathematically, if you have a vector valued function y⃗ =f(x⃗), then the gradient of y⃗  w.r.t x⃗ is the Jacobian matrix.
* Generally speaking, "torch.autograd" is an engine for computing Jacobian-vector product.
* The tensor passed INTO the ".backward(...)" function/method acts like weights for a weighted output of gradient. 
* Mathematically, this is the vector multiplied by the Jacobian matrix of non-scalar tensors; hence it should almost always be a unit tensor of dimension same as the tensor backward is called upon, unless weighted outputs needs to be calculated.
* That is; given any vector: v⃗ = <v1, v2, ..., vn>, "torch.autograd" will compute the product: (J^T * v⃗^T)
* You can think of that vector v⃗ as the vector of "cost/loss" function, while x⃗ (or w⃗) represents the vector of weights/parameters, and y⃗ represents the vector of different outputs depending on how many output nodes exist; with different weights/parameters assigned to each output node
    A Jacobian matrix in very simple words is a matrix representing all the possible partial derivatives of two vectors. It’s the gradient one vector w.r.t another vector.
    If a vector X = x⃗ = <x1, x2, ..., xn> is used to calculate some other vector: f(X) = <f1, f2, ..., fn> through a function f⃗, then the Jacobian matrix (J) contains all the partial derivative combinations of the output vector f⃗ or y⃗ w.r.t to each parameter in X
    
        [∂y1/∂x1    ∂y1/∂x2     ...     ∂y1/∂xn] 
        [∂y2/∂x1    ∂y2/∂x2     ...     ∂y2/∂xn]  
    J = [   .           .       ...         .  ]  Or we can use: ∂f1/∂x1 
        [   .           .       ...         .  ]  
        [∂ym/∂x1    ∂ym/∂x2     ...     ∂ym/∂xn]  
Above matrix represents the gradient of f(X) w.r.t X

* Suppose a PyTorch gradient enabled tensors X as:
    X = <x1, x2, ..., xn> (Let this be the weights/parameters of some DL model)  
    X undergoes some operations to form the output vector Y --> (multiplication by the inputs for eg, or activation function) 
        Y = f(X) = <y1, y2, ..., ym>  --> (Y = output nodes) 
    Y is then used to calculate a scalar loss "l" (∂l/∂Y). Suppose a vector l⃗ happens to be the gradient of the scalar cost/loss "l" w.r.t the vector Y as follows: 
        l⃗ = <∂l/∂y1, ∂l/∂y2, ..., ∂l/∂ym>  --> This is the derivative of the cost/loss function w.r.t to each and every output node (Y) --> (used to find the first entry in the chain rule)
        Eg:
            ∂l/∂y1 = 2 * (l - y1)  OR  2 * (μ - y1)
            
        The vector l⃗ is called the "grad_tensor", and is passed into the ".backward(...)" function as an argument!!!
        
        
* To get the gradient of the loss "l" w.r.t the weights X (∂l/∂X), the Jacobian matrix J is vector-multiplied with the vector l⃗:

                  [∂y1/∂x1    ∂y2/∂x1     ...     ∂ym/∂x1]  [∂l/∂y1]     [∂l/∂x1]  --> Change in total loss produced (from all output nodes) for a change in x1 (input)  
                  [∂y1/∂x2    ∂y2/∂x2     ...     ∂ym/∂x2]  [∂l/∂y2]     [∂l/∂x2]  
    (J^T * v⃗^T) = [   .           .       ...         .  ]  [ .... ]  =  [ .... ]  --> Gives the changes in "loss/cost" of all output nodes w.r.t changes in x1, x2, ..., xn (objective is to minimize overall cost)
                  [   .           .       ...         .  ]  [ .... ]     [ .... ]
                  [∂y1/∂xn    ∂y2/∂xn     ...     ∂ym/∂xn]  [∂l/∂ym]     [∂l/∂xn]
                                    (nxm)                     (mx1)        (nx1)

Where: [∂l/∂x1] = (∂l/∂y1)*(∂y1/dx1) + (∂l/∂y2)*(∂y2/∂x1) + ... + (∂l/∂ym)*(∂ym/∂x1)  --> Find all x1's contributions to the loss function (chain rule)

* One very important step during training cycle is clearing the gradients by calling "zero_grad" pre "backward()" call
'''


##################################################################################################################

x = torch.randn(3, requires_grad=True)
print(f"x = {x}")  # This is a (1x3) (horizontal vector) --> row vector

y = x * 2
n = 1  # To count how many times we multiplied "y" by 2
while y.norm() < 1000:  # ".norm()" is the L2 norm, aka the Euclidean norm (distance) of the tensor/vector --> sqrt(x^2 + y^2 + ... + n^2)
    y *= 2
    n += 1
print(f"y = {y} \nn = {n} \n")

##################################################################################################################
# How "y.norm()" is calculated:
w = torch.randn(3)
d1 = torch.sqrt(torch.sum(torch.pow(w, 2)))  # "torch.pow(tensor, power)" --> raises every element in input tensor to the specified power
d2 = w.norm()
print(d1 == d2, "\n")
##################################################################################################################

# Since tensor "y" is NOT a scalar, ".backward()" cannot compute the gradients, unless we pass a vector of the same dimensions into the ".backward() as an argument:
v = torch.tensor([0.1, 1, 0.001])  # --> Will multiply each respective input in "y" with its corresponding index in "v": [y_1 * 0.1, y_2 * 1, y_3 * 0.01]
y.backward(v)  # [y = 2^n * x] --> [∂y/∂x = 2^n]
print(x.grad)
