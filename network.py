import torch
import torch.nn as nn
import torch.nn.functional as functional

class CharacterRecognizer(nn.Module):

    def __init__(self):
      super(CharacterRecognizer, self).__init__()

      #28 x 28 pools into 14 x 14 then pools into 7 x 7

      # encodinga

      self.conv1 = nn.Sequential(         
        nn.Conv2d(
          in_channels=1,              
          out_channels=32,            
          kernel_size=5,              
          stride=1,                   
          padding=2,                  
        ),                            
        nn.ReLU(),                      
        nn.MaxPool2d(kernel_size=2),    
      )

      self.conv2 = nn.Sequential(         
        nn.Conv2d(
          in_channels=32,              
          out_channels=64,            
          kernel_size=5,              
          stride=1,                   
          padding=2,                  
        ),                               
        nn.ReLU(),                      
        nn.MaxPool2d(kernel_size=2),    
      )

      self.conv3 = nn.Sequential(         
        nn.Conv2d(
          in_channels=64,              
          out_channels=128,            
          kernel_size=5,              
          stride=1,                   
          padding=2,                  
        ),             
        nn.BatchNorm2d(128),                     
        nn.ReLU(),                      
        nn.MaxPool2d(kernel_size=2),    
      )

      # was wrong because there are 47 classes.
      self.out = nn.Linear(3 * 3 * 128, 47)

    def forward(self, x):
      x = self.conv1(x)
      x = self.conv2(x) 
      x = self.conv3(x)
      x = torch.flatten(x, 1)

      x = self.out(x)     
      return x    # return x for visualization


