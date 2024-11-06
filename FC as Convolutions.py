# بسم الله الرحمن الرحيم
# How to make fully connected layers using convolutional layers: https://sebastianraschka.com/faq/docs/fc-to-conv.html
# Can be done by either:
    # 1.) Using a kernel size of same shape and depth as input feature map (f_m), applied "N" time (N=number of outputs)
    # 2.) Using (R*C*depth, #f_m, 1) convolution by reshaping the input feature map shape into (R*C*depth, 1, 1) (Taking image shape into its depth) -> Recall: (Channels, row, column)

import torch

x = torch.tensor([[[[1., 2.],
                    [3., 4.]]]])
print(x.shape, "\n")

fc = torch.nn.Linear(4, 2)  # 4 inputs and 2 outputs
weights = torch.tensor([[1.1, 1.2, 1.3, 1.4],
                        [1.5, 1.6, 1.7, 1.8]])
bias = torch.tensor([1.9, 2.0])

# Randomized weights initially:
print(fc.weight.data, "\n")

# Set weights to our predefined weights:
fc.weight.data = weights
print(fc.weight.data, "\n")
fc.bias.data = bias
y = fc(x.view(-1))  # Flatten the input/image into a *one dimensional* column/row vector
print(y, "\n\n")
'''*** y = X*W.T + b  --> where (y's #rows = #inputs/images) and (y's #columns = #classes/outputs corresponding to each input/image) ***'''


# Method 1:
print("* Using the first method: *")
conv = torch.nn.Conv2d(1, 2, 2)
print(conv.weight.data.shape, "\n")  # --> "(#out_dim, #depth (in_dim), #rows_input, #columns_input)"
conv.weight.data = weights.view(2, 1, 2, 2)
conv.bias.data = bias

y = conv(x)
print(y, "\n\n")

# Method 2:
print("* Using the second method: *")
conv = torch.nn.Conv2d(x[0].numel(), 2, 1)  # "x.numel()" returns the number of entrees in a tensor
print(conv.weight.data.shape, "\n")  # --> "(#out_dim, R*C*depth, 1, 1)" (only 1 feature map (channel) here)
conv.weight.data = weights.view(2, 4, 1, 1)
conv.bias.data = bias

x = x.view(-1, x[0].numel(), 1, 1)  # --> "(1 input/image, 4 channels, 1 row, 1 column)"
y = conv(x)
print(y)
