# بسم الله الرحمن الرحيم

import time
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)  # 2 input nodes mapped to one output node --> * Note: "fc" takes inputs as *ROW* vectors: [x, y, z], not column vectors: [x] *
                                   #                                                                                                                       [y]
                                   #                                                                                                                       [z]
    def forward(self, x):
        x = torch.sigmoid(self.fc(x))  # We didn't use "F.sigmoid()" because it was deprecated
        return x


#                     "l"  "w"
data = torch.Tensor([[[3, 1.5],  # Input data 1
                      [3.5, 0.5],  # Input data 2
                      [4, 1.5],  # Input data 3
                      [5.5, 1]],  # Input data 4

                     [[1, 1],  # Input data 1
                      [2, 1],  # Input data 2
                      [2, 0.5],  # Input data 3
                      [3, 1]]])  # Input data 4

print(f"Data size = {data.size()} \n")  # 2 layers/dimensions, first representing red data, and second is for blue. 4 rows by 2 columns (4x2)

target_r = torch.Tensor([1])  # Arbitrarily, we let red flowers be represented by the number 1, while blue flowers are represented by the number 0  --> Use sigmoid activation function
target_b = torch.Tensor([0])

for r in data[0]:  # Recall that this is taking each "row vector" in the first batch from the "data" tensor
    plt.scatter(r[0], r[1], c="r")
for b in data[1]:
    plt.scatter(b[0], b[1], c="b")
plt.title("Decision Space")
plt.xlabel("Length")
plt.ylabel("Width")
plt.axis([0, 6, 0, 2])
plt.grid()
plt.show()

perf_t = time.perf_counter()  # Counter for benchmarking/setting criteria

net = NN()
print(net, "\n")  # Prints out neural network architecture/specifications

parameters = list(net.parameters())
weights = net.fc.weight.data[0]  # "[0]" enters into the tensor, and ".data" removes the statement: "Parameter containing:"
print(f"Fully Connected Layer weights = {weights} \n")
bias = net.fc.bias.data[0]  # Same as above: "[0]" enters into the tensor and ".data" ==> remove statement
print(f"w1 = {weights[0]} \nw2 = {weights[1]} \nBias = {bias} \n")  # [0] and [1] here are to take single element/item from tensor

criteria = nn.MSELoss()  # Mean Squared Error Loss
losses = []
lr = 1.4  # Try to have initial "lr" as big as possible without breaking the NN  --> Find the tipping point of "lr", in this case its 1.4

for i in range(1000):
    if i == 500:  # Ladder technique: Apply graph oscillation analysis and check where the oscillations tend to reach a stalemate, and then break that by reducing the "lr" to continue descending progress
        lr = 0.45  # This allows us to train our model/algorithm for less epochs, by starting out with a high "lr", thus converging faster initially, then decreasing it to avoid exploding gradient. This saves time during training and increases accuracy.
    elif i == 550:
        lr = 0.435  # If you see oscillations in your "loss" at the end of your epochs, know that your "lr" is too high, because it is bouncing off the minimum loss value!  --> 0.4 is the maximum "lr" in this setting, anything above 0.4 will cause your loss to oscillate and bounce off!
    optimizer = optim.SGD(net.parameters(), lr, momentum=1)
    optimizer.zero_grad()  # Reset the gradient accumulation after completing each epoch
    cost = 0
    for row in data[0]:  # Enter the first batch in "data" tensor, representing the "red" data/sample points
        output = net(row)  # Where "row" represents a row vector = feature vector, with number of columns corresponding to the number of features/inputs (2 in this case)
        loss = criteria(output, target_r)
        cost += loss
        loss.backward()  # Find the gradient of the loss function w.r.t weights
    for row in data[1]:  # Enter the second batch in "data" tensor, representing the "blue" data/sample points
        output = net(row)
        loss = criteria(output, target_b)
        cost += loss
        loss.backward()  # *Accumulate* all the gradients of the loss function w.r.t every weight (every sample point)
    losses.append(cost.item())  # Append cost/loss value after every epoch
    optimizer.step()  # Does the optimization process (updates all the parameters) after we computed and accumulated the gradient of the loss for EVERY sample point ==> Batch Gradient Descent


plt.plot(losses, c="m")
plt.title("Batch Gradient Descent")
plt.xlabel("Epochs")
plt.ylabel("Loss", rotation=0)
plt.grid()

print(f"Optimized_w1 = {weights[0]} \nOptimized_w2 = {weights[1]} \nOptimized bias = {bias} \n")
print(f"∂l/∂w1 = {net.fc.weight.grad[0][0]} \n∂l/∂w2 = {net.fc.weight.grad[0][1]} \n∂l/∂b = {net.fc.bias.g_T[0]} \n\n")  # We add the 1st "[0]" to access the *list*, then the 2nd "[0]" access the element in the tensor

x_red = torch.Tensor([3, 1.25])  # This point is barely in the "red" zone; it's very close to the decision boundary (~1)
y_red = torch.sigmoid(weights[0]*x_red[0] + weights[1]*x_red[1] + bias)
print(f"Our prediction for red says = {y_red}")

x_blue = torch.Tensor([3, 0.75])  # This point is barely in the "blue" zone; it's very close to the decision boundary (~0)
y_blue = torch.sigmoid(weights[0]*x_blue[0] + weights[1]*x_blue[1] + bias)
print(f"Our prediction for blue says = {y_blue} \n")


print(f"Performance Time = {time.perf_counter()-perf_t:.4} seconds")
plt.show()

'''
To specify a specific learning rate for specific layers or a specific parameter:
    Create a dictionary within a list in the optimizer that contains "params" as a key with the specific model's parameters as its key value,
    and also contains "lr" as a key with the desired learning rate as its key value.
    You will need to create a unique dictionary for each set of parameters that require a different learning rate value.
    For the base parameters (that don't require a specific "lr"), you define the "params" key without the "lr" key in the dictionary! 
    And hence, it  will use the "global" lr specified at the end of the optimizer argument.
    
    (*** NOTE ***: All the model's parameters MUST be specified in order to be optimized/trained!!!)
Eg:

exclude = ["conv1.weight", "conv1.bias", "custom_param"]  # Parameters we want to have specific learning rates for (exclude from global "lr")
base_params = [p[1] for p in model.named_parameters() if p[0] not in exclude]  # Where "p[1]" resembles the "Parameter containing" entry (weights and biases), while "p[0]" resembles the parameter's name (string) entry
optimizer = optim.Adam([{"params": base_params}, {"params": model.conv1.parameters(), "lr": alpha*0.01}, {"params": model.custom_param, "lr": alpha*0.1}], lr=alpha)
'''
