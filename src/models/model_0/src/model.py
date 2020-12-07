import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



class Create(torch.nn.Module):
    def __init__(self, input_shape, outputs_count):
        super(Create, self).__init__()

        self.device = "cpu"

        fc_width = input_shape[1]//(2**4)
        
        self.layers = [     
            nn.Conv1d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
           
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
    
          
            Flatten(),
            nn.Linear(128*fc_width, outputs_count[0])
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
        torch.save(self.model.state_dict(), path + "./model.pt")

    def load(self, path):       
        self.model.load_state_dict(torch.load(path + "./model.pt", map_location = self.device))
        self.model.eval()  
    

if __name__ == "__main__":
    input_shape = (3, 1024)
    batch_size  = 16

    model = Model(input_shape, 3)

    x = torch.randn((batch_size, ) + input_shape)
                    
    y = model.forward(x)
    
    print("y shape = ", y.shape)
    print(y)