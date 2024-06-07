import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, input_channels, num_layers, layer_params, embedding_dim):
        super(CNNEncoder, self).__init__()
        
        layers = []
        in_channels = input_channels
        
        for i in range(num_layers):
            out_channels = layer_params[i]['out_channels']
            kernel_size = layer_params[i]['kernel_size']
            stride = layer_params[i]['stride']
            padding = layer_params[i]['padding']
            
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            )
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            in_channels = out_channels
            
        self.conv_layers = nn.Sequential(*layers)
        
        # Assuming the final output of conv_layers is of shape (batch_size, out_channels, 1, 1)
        self.embedding = nn.Linear(out_channels, embedding_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.embedding(x)
        return x    
    
    def l2_loss(self, x1, x2):
        return F.mse_loss(x1, x2)
    
    def bce_loss(self, predictions, targets):
        loss = F.binary_cross_entropy_with_logits(predictions, targets)
        return loss

class Predictor(nn.Module):
    def __init__(self, embedding_dim, num_hidden_layers, hidden_layer_sizes):
        super(Predictor, self).__init__()
        
        layers = []
        in_features = embedding_dim
        
        for i in range(num_hidden_layers):
            hidden_dim = hidden_layer_sizes[i]
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        
        layers.append(nn.Linear(in_features, embedding_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x