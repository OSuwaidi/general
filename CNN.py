# بسم الله الرحمن الرحيم
"""
* Constructing Neural Networks using the "torch.nn" ("as nn") package.
* A "nn.Module" is a *class* that contains layers (attributes) to be called in the method called "forward(input)" that returns the "output".
      For example, "nn.Conv2d" is a method that will take in (in_channels, out_channels, kernel (nxn), stride, padding) and apply convolution on an image array

* A typical training procedure for a neural network model is as follows:
- Define the neural network (architecture) that has some learnable parameters (weights)
- Iterate over a dataset of inputs/batches (epochs)
- Process input through the network (feedforward)
- Compute the loss (how far is the output from being correct (predicted - actual))
- Propagate gradients back into the network’s parameters (backpropagation)
- Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient (optimization)

* Note:
- "torch.nn" only supports mini-batches  ==> Have to divide the labeled dataset into batches/parts (helps w/ cross-validation)
- The entire "torch.nn" package ONLY supports inputs that are a mini-batch of samples, and not a single batch of samples.
- If you have a single sample, just use "input.unsqueeze(0)" to add a fake batch dimension.

* Note:
- "nn.Linear" allows us to specify input nodes and output nodes respectively; then it automatically multiplies every input node/feature by a random set of weights/parameters and fully connects them to every possible output node!

* TLDR: (https://discuss.pytorch.org/t/beginner-should-relu-sigmoid-be-called-in-the-init-method/18689)
- In the standard use case, you are writing some kind of model with some layers. The layers hold most likely some parameters which should be trained.
  "nn.Conv" would be an example; and these layers that contain trainable parameters are created under "__init__" (in "nn.Module").
  On the other hand, some layers don’t have any trainable parameters like "nn.MaxPool".
  Other 'layers' that don’t have parameters, which can be seen as simple functions instead of proper layers like: "nn.ReLU" are created in the "forward" method
  In your "forward" method you are creating the *logic* of your forward (and thus also backward) pass.
  In a very simple use case, in your "forward" method, you would just call all created layers one by one, passing the output of one layer into the other.
  However, since the computation graph is created dynamically, you can also create some crazy forward passes, e.g: using a random number to select a repetition of some layers, split your input into different parts and call different “sub modules”, skip a layer or some nodes in a layer, etc. You are not bound to any convention during this, as long as all shapes match.
  This is one of the most beautiful parts of PyTorch. You can let your imagination flow without worrying too much about some limited API which can only call layers in a sequential manner.
      - Basically, "nn.Module" = architecture: you define attributes, methods, and *learnable parameters*, whereas in the "forward" method, you actually use and IMPLEMENT those defined methods/attributes and learnable parameters!

* Note:
    'Columns' represent the number of features of the input and/or output nodes, while 'rows' represent the number of data/sample points
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # Activation functions package


# 1.) Defining a neural network:


class Net(nn.Module):  # "nn.Module" is a (super/parent) class that contains several methods and attributes that are useful for defining a neural network
    def __init__(self):  # Recall: "__init__" initializes the module including its parameters in each forward pass
        super().__init__()  # *Note: we can't define "init" under a child class, thus: "super" --> Makes it (Net) a super/parent class*. If the actual parent class requires arguments, then you would insert then inside the "init" as such: "super().__init__(*arguments)"
        # Attributes:
        # 1 input image channel (gray scale), 6 output channels, 3x3 square convolution kernel (since square matrix we can just specify a single number "3")
        self.conv1 = nn.Conv2d(1, 6, 3)  # --> "nn.Conv2d(in_channels, out_channels, kernel (3x3), stride, padding)"  --> "stride": how many pixels to skip/shift over when convolving, "padding": what are the values assigned to the pixels outside of the image range (default=0)
        self.conv2 = nn.Conv2d(6, 16, 3)  # --> Applies a 2D convolution over an input signal (image) composed of several input channels/planes.
        # "Linear" = an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*(6*6), 120)  # --> "nn.Linear(in_features, out_features)"  --> 16 came from number of channels of final layer, and (6x6) came from applying the (3x3) convolutions on the original 32x32 image twice, then applying a (2x2) maxpooling layer twice ==> [conv1: -2 (30x30)] -> [maxpool1: /2 (15x15)] -> [conv2: -2 (13x13)] -> [maxpool2: /2 (6x6)]
        self.fc2 = nn.Linear(120, 84)  # --> "fc" = fully connected layer: inputs to outputs (automatically multiplies every input with corresponding weights/parameters to form "84" output nodes (in this case))
        self.fc3 = nn.Linear(84, 10)  # --> Applies a linear transformation onto the incoming data: y = w*x + b

    # Static method as it takes no attributes from the "__init__" constructor above (doesn't use "self"):
    @staticmethod
    def num_flat_features(x):  # Flatten the input image into a row vector after convolving and maxpooling to pass through our NN
        size = x.shape[1:]  # All dimensions except the "batch" dimension  --> (channels, height, width) --> (16, 6, 6)
        num_features = np.prod(size)
        return num_features  # Technically since you flattened the image into (1xn) row vector, this "n" becomes "num_features"

    def forward(self, x):  # "x" = input image to be processed
        # Max pooling over a (2, 2) window:  (*Max pooling has a stride of 2 by default*)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # Could have just specified a single number "2"
        # If the size is a square (eg: 2x2) you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # "Max Pooling" is used to reduce the size/dimensionality of our input data, by only taking the maximum pixel value in a given block of an image, and thus reducing the image down to those maximum pixel values (keeping/preserving the most "important" features of an image) -> Thus, making the model SHIFT INVARIANT!!!
        x = x.view(-1, self.num_flat_features(x))  # --> Could have used: "x.view(1, -1)"  --> We flatten the final layer produced by the CNN (2D to 1D layer) for passing into the flat layer (fc) of neurons (NN) as a *row vector*  --> # columns = # features
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


'''
*IMPORTANT*: In PyTorch, the output is produced by the following in matrix notation: 
             y = X*W.T + b  (inputs come first then weights)
             Where (y's #rows = #inputs/images) and (y's #columns = #classes/outputs corresponding to each input/image)
* Note:
    Instead of using the stateless object "F.relu" directly from the "functional" package/API, we can define "relu"
    under our module's "init" constructor, so that we can try different activation functions in our forward pass,
    without repeatedly having to change the "F.relu" function to another non-linear activation function everytime.

    - Eg:
        class MyModel(nn.module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)  --> (in_channels=3, out_channels=6, (3x3), stride=1, padding=1)  --> Specifying one dimension for size in "Conv2d" implies a square matrix
                self.act = nn.ReLU()  --> Can just change "nn.ReLU" to any other activation function we want, and run the code as usual

            def forward(self, x):
                x = self.act(self.conv1(x))
                return x


* Note: 
    If in some case you were using a conv layer, but for whatever reason you needed to access and manipulate its weight often. 
    The usual approach is to create the conv layer in "__init__", then applying it in "forward". 
    However, accessing the conv weights might be a bit troublesome this way.
    So how about we just store the filter (convolution) weights as "nn.Parameters" in "__init__" and just use the functional API "F.conv2d" in the forward "method". 
    Since we’ve properly registered the filter weights in "__init__", they will be trained as long as they are used somewhere in the computation during the forward pass.

    - Eg:
        class MyModel(nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                conv1 = nn.Conv2d(1, 6, 3)
                self.weight = conv1.weight
                self.bias = conv1.bias

            def forward(self, x):
                x = F.conv2d(x, self.weight, self.bias)
                return x


        model = MyModel()
        x = torch.randn(1, 1, 4, 4)  --> Input image
        out = model(x)  --> Process image (forward pass)


* Note:
    If we were to use the convolutional layer "nn.Conv2d" in the "forward" method as such: 

    def forward(self, x):
        x = nn.Conv2d(3, 6, 3)(x) --> Wrong!
        x = self.act(x) 
        return x  

    Then everytime you pass an input image (from different batches) into your "net()" class object, the learnable parameters (of the convolutional layer) would be reinitialized and overwritten by the next forward pass!
'''

net = Net()
print(net, "\n")  # Prints the specifications of our NN model

# You just have to define the "forward" function... then the "backward" function (where gradients are computed) is automatically defined for you using "autograd"
# You can use any of the Tensor operations in the forward pass
# The learnable parameters of a model are returned by "net.parameters()"
params = list(net.parameters())  # Contains ALL the parameters/weights used in our CNN
print(f"Number of output parameters = {len(params)}")  # Prints the length/number of the *output* parameters
print(params[0].size())  # "[0]" refers to weights assigned on the first layer, i.e: conv1's weights: 6 batches of (3x3) matrix of weights/parameters == 54 weights  --> *Note: the (3x3) kernel parameters are the same for the same layer!*
print(f"weights = {params[0]} \n")  # Since our "Conv2d" is a (3x3) kernel and we have 6 outputs, we get 6 batches of parameters, each with different wights, assigned for different outputs

# 2.) Processing inputs:

# Let’s try a random 32x32 input. Note: expected input size of this net (LeNet) is 32x32. To use this net on the MNIST dataset, we need to resize the images from the dataset to 32x32.
inpt = torch.randn(1, 1, 32, 32)  # --> "torch.randn(# batches, # channels, height, width)"  --> "torch.randn(N, C, H, W)"
output = net(inpt)  # We passed our input image: "inpt" (x), into the object/instance "net"
print(f"out = {output} \n")  # --> (1x10) row vector: 10 outputs resembling the 10 digits: 0 through 9  --> # columns == # outputs

# 3.) Calling backward:

# Set gradients of all model parameters to zero:
net.zero_grad()
# Then backpropagate with random gradients:
output.backward(torch.randn(1, 10))  # 10 random gradients, for 10 output nodes

############################################################################################################
# 4.) Computing the loss:
'''
* A loss function takes the (output, target/actual) as a pair of inputs, and computes a value that estimates how far away the output is from the target.
* There are several different loss functions under the "nn" package . A simple loss is: "nn.MSELoss" which computes the mean-squared error between the input and the target.
'''
output = net(inpt)
target = torch.rand(1, 10)  # Toy target as a (1x10) row vector --> make it same shape as output
criteria = nn.MSELoss()

loss = criteria(output, target)
print(f"loss/cost = {loss} \n")

# Following "loss" to see what operations lead to it's formation gives a computational graph as follows:
# Forward pass: conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d -> view/flatten -> linear -> relu -> linear -> relu -> linear -> MSELoss -> loss
# Backward pass: conv2d <- relu <- maxpool2d <- conv2d <- relu <- maxpool2d <- view <- linear <- relu <- linear <- relu <- linear <- MSELoss <- loss
print(loss.grad_fn)  # MSELoss operation (loss function)
print(loss.grad_fn.next_functions[0][0])  # Linear operation (FC layer)
print(loss.grad_fn.next_functions[0][0].next_functions[0][0], "\n")  # ReLU operation (activation function)

# Calling "loss.backward()" computes all the derivatives until the leaf nodes, and all Tensors that had "requires_grad=True" will have their ".grad" Tensor accumulated with the gradient.
# To backpropagate the error all we have to do is "loss.backward()". We need to clear the existing gradients though, else gradients will be accumulated to existing gradients.
net.zero_grad()

# Find conv1’s bias gradients before and after backprop (∂loss/∂b1): (where b1 represents the array of biases of the first layer in "conv1")
print(f'conv1.bias.grad before backward = {net.conv1.bias.grad}')
loss.backward()
print(f'conv1.bias.grad after backward = {net.conv1.bias.grad} \n')
# To find conv1's weights gradients (∂loss/∂w1) (where w1 represents the matrix of weights of the first layer "conv1") --> "net.conv1.weight.grad"


# 5.) Updating the weights of the network:

# The simplest update rule used in practice is the Stochastic Gradient Descent (SGD):
# weight = weight - learning_rate * gradient
learning_rate = 0.01
for w in net.parameters():
    w.detach().sub(learning_rate * w.grad)  # "w.sub(x)" --> Subtracts tensor "x" from tensor "w"; element-wise operation
# *** Note ***: A leaf Variable that requires grad CANNOT be used in an in-place operation!!!  --> Therefore, we had to detach leaf variables "w" (parameters) to be able to do an in-place operation, and alter the original parameters


############################################################################################################
# However, as you use neural networks, you want to use various different update rules such as: SGD, Nesterov-SGD, Adam, RMSProp, etc.
# To enable this, we use a small package: "torch.optim" that implements all these methods:
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01)  # Create the optimizer to optimize our parameters

optimizer.zero_grad()  # zero out the gradients ("optimizer.zero_grad()" has no effect if we have a single ".backward()" call, as the gradients are already zero to begin with (technically "None" but they will be automatically initialised to zero))
# Note that you have to zero out the gradients because gradients are accumulated, i.e: they will be accumulated to the existing/previous gradients

output = net(inpt)
loss = criteria(output, target)
loss.backward()
optimizer.step()  # Applies a step in gradient descent to update the parameters of our model (to be inside of a "for" loop)
