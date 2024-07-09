"""
Definitions of simple dnn models.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()

        self.fc1 = nn.Linear(784, 512) 

        self.fc2 = nn.Linear(512, 256)  

        self.fc3 = nn.Linear(256, 10)  

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.zeros_(self.fc2.bias)
        nn.init.normal_(self.fc3.weight, std=0.01)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):

        x = x.view(-1, 784)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x

class SimpleDNN_2(nn.Module):
    def __init__(self, other):
        super(SimpleDNN_2, self).__init__()

        self.fc1 = nn.Linear(784, 512)  

        self.fc2 = nn.Linear(512, 256)  
        self.up = nn.Linear(512,10)
        self.down = nn.Linear(10,256)

        self.fc3 = nn.Linear(256, 10)   
        self.init_weights(other)

    def init_weights(self, other):
        self.fc1.weight.data = other.fc1.weight.data
        self.fc1.bias.data = other.fc1.bias.data
        self.fc2.weight.data = other.fc2.weight.data
        self.fc2.bias.data = other.fc2.bias.data
        self.fc3.weight.data = other.fc3.weight.data
        self.fc3.bias.data = other.fc3.bias.data
        nn.init.normal_(self.up.weight, std=0.01)
        nn.init.uniform_(self.up.bias)
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.uniform_(self.down.bias)


    def forward(self, x):
        x = x.view(-1, 784)

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x)+self.down(self.up(x)))

        x = self.fc3(x)

        return x


model = SimpleDNN()

other_model = []
for i in range(2,6):
    other_model.append(SimpleDNN_2(model))

model.eval()

for m in other_model:
    m.eval()

dummy_input = torch.randn(1, 784)

torch.onnx.export(model,
                  dummy_input,
                  "./model/model_1.onnx",
                  export_params=True,
                  opset_version=10,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})
for idx, m in enumerate(other_model,2):
    torch.onnx.export(m,
                    dummy_input,
                    f"./model/model_{idx}.onnx",
                    export_params=True,
                    opset_version=10,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})

# torch.onnx.export(model_2,
#                   dummy_input,
#                   "model_2.onnx",
#                   export_params=True,
#                   opset_version=10,
#                   input_names=['input'],
#                   output_names=['output'],
#                   dynamic_axes={'input': {0: 'batch_size'},
#                                 'output': {0: 'batch_size'}})