import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, weight_init_gain = 1.0):
        super(ResidualBlock, self).__init__()

        self.conv0  = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act0   = nn.ReLU()
        self.conv1  = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.act1   = nn.ReLU()
            
        torch.nn.init.xavier_uniform_(self.conv0.weight, gain=weight_init_gain)
        torch.nn.init.xavier_uniform_(self.conv1.weight, gain=weight_init_gain)

    def forward(self, x):
        y  = self.conv0(x)
        y  = self.act0(y)
        y  = self.conv1(y)
        y  = self.act1(y + x)
        
        return y


class Model(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 64):
        super(Model, self).__init__()

        self.device = "cpu"

        fc_size = hidden_count*input_shape[1]*input_shape[2]
        
        self.layers = [     
            nn.Conv2d(input_shape[0], hidden_count, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            ResidualBlock(hidden_count),
            ResidualBlock(hidden_count),
            ResidualBlock(hidden_count),
            ResidualBlock(hidden_count),
            ResidualBlock(hidden_count),
            ResidualBlock(hidden_count),
            ResidualBlock(hidden_count),
            ResidualBlock(hidden_count),
  
            Flatten(),

            nn.Linear(fc_size, 64),
            nn.ReLU(),
            nn.Linear(64, outputs_count)
        ]

        for i in range(len(self.layers)):
            if hasattr(self.layers[i], "weight"):
                torch.nn.init.xavier_uniform_(self.layers[i].weight)


        self.model = nn.Sequential(*self.layers)
        self.model.to(self.device)

        print("model")
        print(self.model)
        print("\n\n")
       

    def forward(self, x):
        return self.model(x)

     
    def save(self, path):
        torch.save(self.model.state_dict(), path + "trained/model.pt")

    def load(self, path):       
        self.model.load_state_dict(torch.load(path + "trained/model.pt", map_location = self.device))
        self.model.eval()  
    

if __name__ == "__main__":
    input_shape = (3, 38, 50)
    batch_size  = 16

    model = Model(input_shape, 3)

    x = torch.randn((batch_size, ) + input_shape)
                    

    y = model.forward(x)
    
    print("y shape = ", y.shape)
    print(y)