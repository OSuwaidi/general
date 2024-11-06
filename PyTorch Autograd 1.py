# بسم الله الرحمن الرحيم
"""
* Gradient accumulation refers to the situation, where 'multiple' backwards passes are performed before updating the parameters. The goal is to have the same model parameters for multiple "sets of inputs" (batches) and then update the model's parameters based on all these batches, instead of performing an update after every single batch  --> Call backward multiple times, then optimize once.
* Gradient accumulation means running a configured number of steps without updating the model variables while accumulating the gradients of those steps and then using the accumulated gradients to compute the variable updates.
* Running some steps without updating any of the model variables is the way we (logically) split the batch of samples into a few mini-batches. The batch of samples that is used in every step is effectively a mini-batch, and all of the samples from those steps combined are effectively the global batch.
    TLDR: We use gradient accumulation to pass our training dataset in batches, instead of all at once (a big chunk)
          And at each batch, we compute the gradient of our loss function (backprop), then by accumulating those gradients (step), we add all those loss function gradients together to get the "global" loss function gradient
          Then we PASS/propagate that "global" loss function gradient back into our NN via backpropagation to update our weights/parameters using gradient descent
          *** Accumulating the gradients in all of these steps (min-batches) results in the SAME sum of gradients as if we were using the global batch size!!! ***  --> This is true because the cost/loss function is the *sum* of: (predicted - target)**2 of all datasets, and since the different losses are separated by a "+" sign, their derivatives/gradients and "independent" from one another!
    Eg:
        We are accumulating gradients over 5 steps. We want to accumulate the gradients of the first 4 steps, without updating any variable. At the fifth step, we want to use the accumulated gradients of the previous 4 steps combined with the gradients of the fifth step to compute and assign the variable updates.

* The previous computational graph is automatically destroyed when another ".backward()" is called, however the previous gradients remain in memory (to be accumulated)
* To erase previously computed gradients (accumulated gradients) we need to explicitly call "zero_grad()"
* (Maybe try to update the parameters after every batch to see if that gives better generalization of out data, though I doubt it)
"""
import torch

a = torch.randint(0, 10, (2, 3))
b = torch.randint(10, (4, 3))
print(f"a = {a} \nb = {b} \n")

# To concatenate:
c = torch.cat([a, b])  # Combine by stacking row-wise  (have to be of same column-size!!!)
print(f"μ = {c} \nc_dim = {c.size()} \n")  # Dimensions: 2 + 4 rows = 6 rows. Thus (6x3) matrix

at = a.T  # (3x2)
bt = b.T  # (3x4)
print(f"at = {at} \nbt = {bt} \n")

ct = torch.cat([at, bt], 1)  # Combine by stacking column-wise  (have to be of same row-size!) ("1" specifies axis)
print(f"ct = {ct} \nct_dim = {ct.size()}\n\n")  # Dimensions: 2 + 4 columns = 6 columns. Thus (3x6) matrix


# Using the ".view()" method to reshape a tensor. This method receives heavy use, because many neural network components expect their inputs to have a certain shape:
x = torch.randn(2, 3, 4)  # --> ".randn(dim, r, μ)"  --> 24 elements: 2*(3x4) = 2*(12) = 24
print(f"x = {x} \nx_dim = {x.size()} \n\nx_re = {x.view(2, 12)}\n")  # Reshape 3D tensor "x" from: 2*(3x4) dimensions to (2x12) dimensions (2D tensor)

# The following is same as above. If one of the dimensions is "-1", this implies that its size can be inferred/interpreted (determined based on first input size to make it match)
print(x.view(2, -1), "\n\n")

'''
***** Computation Graphs and Automatic Differentiation *****

# Gradients are calculated by tracing the graph from the roots (outputs) to the leaves (inputs/leaf-nodes) and multiplying every gradient in the way using the chain rule.
# "is_leaf": A node is *leaf* if:
    * It was initialized explicitly by some function like "x = torch.tensor([[1.0]])" or "x = torch.randn(1, 1)".
        μ = torch.rand(10, requires_grad=True) + 2
        μ.is_leaf = False  --> "μ" was created by the addition operation!
    * It is created after operations on tensors which all have "requires_grad" = False.
    * It is created by calling ".detach()" method on some tensor.
# When an output tensor is created by any means of arithmetic operation, it has no information on HOW it was created (all this output tensor knows is its own data and shape/size only)
# To have the Tensor object keep track of how it was created (of what previous matrix operations was it generated) --> set the flag to "requires_grad=True"
# *** NOTE *** : Only Tensors of floating point and complex dtype can require gradients, integer tensors CANNOT! --> "torch.tensor(FLOAT_TENSOR, requires_grad=True)"
# Also note that "torch.arange()" gives out an int64 data type input, while "torch.linspace()" gives a float32 data type input. Therefore, we could use "require_grad" when using "torch.linspace()", but not "torch.arange()"
# Actually we could still use the "arange" method by specifying the data type: "torch.arange(i, f, dtype=torch.float32)
'''
x = torch.tensor([1., 2., 3], requires_grad=True)  # The flag here is "requires_grad"!
y = torch.tensor([4., 5., 6], requires_grad=True)
z = x + y
w = y - x
print(f"z = {z}")
# But "z" knows something extra!
print(f"z_info =  {z.grad_fn} \n")  # Location in memory where this extra information is stored
print(f"w = {w} \nw_info = {w.grad_fn}\n ")  # Thus now Tensors know what created them and how

# Q.) But how does this help us in computing the gradient/derivative?
s = z.sum()  # The sum of all elements in tensor "z"
print(f"s = {s}")  # "s" here is the output node
print(s.grad_fn)

'''
So now, what is the derivative of this sum "s" w.r.t the first component in "x" (x0)? --> ∂s/∂x0
"s" knows that it was created as a sum of the tensor "z". "z" knows that it was the sum of "x + y", therefore:

z0 = x0 + y0
z1 = x1 + y1
z2 = x2 + y2
z = [z0, z1, z2]

s = z0 + z1 + z2 
s = (x0 + y0) + (x1 + y1) + (x2 + y2)
∂s/∂x0 = (1 + 0) + 0 + 0

Now taking the derivative of tensor "s" w.r.t variable vector "x" (not w.r.t any specific one): 
∂s/∂x = (1 + 0) + (1 + 0) + (1 + 0)
'''
# Now to actually compute the gradient via back propagation ***(OUTPUT TENSOR HAS TO BE SCALAR TO BE ABLE TO BACKPROP)***:
s.backward()  # Note: calling ".backward()" on any variable will run back propagation, STARTING from it
print(f"∂s/∂x = {x.grad} \n\n")  # Gradient/derivative of "s" w.r.t variable "x"  -->  gives [∂s/∂x0] and [∂s/∂x1] and [∂s/∂x2]


'''
".backward()" is the function which actually calculates the gradient by passing it’s argument (1x1 unit tensor by default) through the backward graph all the way up to every leaf node traceable from the calling root tensor. 
The calculated gradients are then stored in ".grad" of every LEAF node. Remember, the backward graph is already made dynamically during the forward pass!
Thus, backward function only calculates the gradients using the already made computational graph, and then stores them in leaf nodes as ".grad".

On calling "backward()", gradients are populated ONLY for the nodes which have BOTH "requires_grad" and "is_leaf" True. 
Gradients are calculated from the output node from which ".backward()" is called, w.r.t other leaf nodes.
'''
# Eg:
x = torch.tensor(1., requires_grad=True)  # Leaf node
z = x**3  # Output node
z.backward()  # Apply backpropagation starting from output (z) to input (x) (applying chain rule)
print(f"dz/dx = {x.grad} \n")  # Call the attribute ∂z/∂x

# Note: The dimension of tensor passed into ".backward()" must be the same as the dimension of the tensor whose gradient is being calculated. (It is set to "torch.tensor(1.)" be default)
# For example, if the gradient enabled tensor "x" and "y" are as follows:
x = torch.tensor([0., 2, 8], requires_grad=True)  # --> specifying any element as a float, makes the entire tensor of type float32
y = torch.tensor([5., 1, 7], requires_grad=True)
z = x * y
# then, to calculate gradients of "z" (a 1x3 tensor) w.r.t "x" or "y" , an external gradient needs to be passed to "z.backward()" function as follows:
z.backward(torch.Tensor([1, 1, 1]))
print(f"∂z/∂x = {x.grad} \n\n")  # ∂z/∂x = 1 * y


'''
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
    X undergoes some operations to form the vector Y --> (multiplication by the inputs for eg, or activation function) 
        Y = f(X) = <y1, y2, ..., ym>  --> (Y = output nodes) 
    Y is then used to calculate a scalar loss "l" (∂l/∂Y). Suppose a vector l⃗ happens to be the gradient of the scalar cost/loss "l" w.r.t the vector Y as follows: 
        l⃗ = <∂l/∂y1, ∂l/∂y2, ..., ∂l/∂ym>  --> This is the derivative of the cost/loss function w.r.t to each and every output node (Y) --> (used to find the first entry in the chain rule)
        Eg:
            ∂l/∂y1 = 2 * (l - y1)  OR  2 * (μ - y1)
            
        The vector l⃗ is called the "grad_tensor", and is passed into the ".backward(...)" function as an argument!!!
        
        
* To get the gradient of the loss "l" w.r.t the weights X (∂l/∂X), the Jacobian matrix J is vector-multiplied with the vector l⃗:

                  [∂y1/∂x1    ∂y2/∂x1     ...     ∂ym/∂x1]  [∂l/∂y1]     [∂l/∂x1]  --> Change in total loss produced (from all output nodes) for a change in x1 (input)  
                  [∂y1/∂x2    ∂y2/∂x2     ...     ∂ym/∂x2]  [∂l/∂y2]     [∂l/∂x2]  
    (J^T * l⃗^T) = [   .           .       ...         .  ]  [ .... ]  =  [ .... ]  --> Gives the changes in "loss/cost" of all output nodes w.r.t changes in x1, x2, ..., xn (objective is to minimize overall cost)
                  [   .           .       ...         .  ]  [ .... ]     [ .... ]
                  [∂y1/∂xn    ∂y2/∂xn     ...     ∂ym/∂xn]  [∂l/∂ym]     [∂l/∂xn]
                                    (nxm)                     (mx1)        (nx1)

Where: [∂l/∂x1] = (∂l/∂y1)*(∂y1/dx1) + (∂l/∂y2)*(∂y2/∂x1) + ... + (∂l/∂ym)*(∂ym/∂x1)  --> Find all x1's contributions to the loss function (chain rule)

* One very important step during training cycle is clearing the gradients by calling "zero_grad" pre the ".backward()" call
'''


##################################################################################################################
x = torch.randn(2, 2)
y = torch.randn(2, 2)
# By default, user created Tensors have "requires_grad=False"
print(x.requires_grad, y.requires_grad)  # --> Both False
z = x + y

# Therefore you can't backprop through z
print(f"Gradient info = {z.grad_fn} \n")  # Not stored in any memory location


# ".requires_grad_()" changes an existing Tensor's "requires_grad" flag in-place. The input flag defaults to "True" if not given an argument.
x = x.requires_grad_()  # --> True
y = y.requires_grad_()  # --> True
# z now contains enough information to compute gradients, as we saw above:
z = x + y

print(f"Gradient info = {z.grad_fn}")
# If any input to an operation has "requires_grad=True", so will the output
print(z.requires_grad, "\n")  # --> True

# Now z has the computation history that relates itself to "x" and "y"
# Can we just take its values, and **detach** it from its history?
# "detach()": detaches the output from the computational graph
new_z = z.detach()  # "new_z" is now a leaf tensor because we detached it!  --> Similarly, we could've used "new_z = z.data" (worse idea) --> These operations ONLY capture the data stored, not the history of operations

# Does the "new_z" have information to backprop to "x" and "y"?
# NO!
print(f"Gradient info = {new_z.grad_fn} \n\n")
# And how could it? "z.detach()" returns a tensor that shares the same data/storage as the ORIGINAL "z" only, but with the computation history forgotten --> (requires_grad=False)!!! It doesn't know anything about how it was computed
# In essence, we have broken the Tensor away from its past history


# Creating the graph:
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0)
z = x * y

# Displaying:
for tensor, name in zip((x, y, z), "xyz"):  # --> "tensor" calls (x, y, z), while "name" calls from the string "xyz"
    print(f"{name} \ndata: {tensor.data} \nrequires_grad: {tensor.requires_grad} \ngrad: {tensor.grad} \ngrad_fn: {tensor.grad_fn} \nis_leaf: {tensor.is_leaf}\n")


'''
* You can also stop autograd from tracking history on Tensors with ".requires_grad=True" by wrapping the code block in with "torch.no_grad()":

x = torch.tensor(1.0, requires_grad = True)
print(x.requires_grad)  --> True
y = x * 2
print(y.requires_grad) --> True

with torch.no_grad():
    y = x * 2
    print(y.requires_grad) --> False
  
    
* Or by using ".detach()" to get a new Tensor with the same content/data but that does not require gradients:
x = torch.tensor(1.0, requires_grad = True)
print(x.requires_grad)  --> True
y = x.detach()
y = x * 2
print(y.requires_grad) --> False
'''