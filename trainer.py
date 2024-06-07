import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import CNNEncoder, Predictor
from data_class import MNISTCSVDataset

class Trainer:
    def __init__(self, config):
        self.config = config

        # Initialize networks
        self.encoder = CNNEncoder(**config['encoder'])
        self.lagging_encoder = CNNEncoder(**config['encoder'])
        self.predictor = Predictor(**config['predictor'])

        # Copy initial weights from encoder to lagging_encoder
        self.lagging_encoder.load_state_dict(self.encoder.state_dict())

        # Define optimizer
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=config['lr'])
        
        # EMA decay factor
        self.ema_decay = config['ema_decay']

    def update_ema(self):
        for param, ema_param in zip(self.encoder.parameters(), self.lagging_encoder.parameters()):
            ema_param.data = self.ema_decay * ema_param.data + (1 - self.ema_decay) * param.data

    def train_epoch(self, dataloader):
        self.encoder.train()
        self.predictor.train()
        self.lagging_encoder.eval()
        
        total_loss = 0
        
        for batch in dataloader:
            img, masked_img, _ = batch
            img, masked_img = img.to(self.config['device']), masked_img.to(self.config['device'])

            # Forward pass
            masked_embedding = self.encoder(masked_img)
            img_embedding = self.lagging_encoder(img)
            predicted_embedding = self.predictor(masked_embedding)
            
            # Compute loss
            loss = self.encoder.l2_loss(predicted_embedding, img_embedding)
            total_loss += loss.item()
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update EMA of lagging encoder
            self.update_ema()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def train(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(dataloader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Configuration for initializing networks
config = {
    'encoder': {
        'input_channels': 1,
        'num_layers': 3,
        'layer_params': [
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ],
        'embedding_dim': 288
    },
    'predictor': {
        'embedding_dim': 288,
        'num_hidden_layers': 2,
        'hidden_layer_sizes': [256, 256]
    },
    'lr': 0.001,
    'ema_decay': 0.99,
    'device': torch.device('cpu' if torch.cuda.is_available() else 'cpu')
}

# Create dataset and dataloader
csv_file = '/home/george-vengrovski/Documents/studying/cs_572/final_project/dev.csv'  # Update this path to your CSV file
dataset = MNISTCSVDataset(csv_file, mask_width=7)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize trainer and train
trainer = Trainer(config)
trainer.train(dataloader, num_epochs=10)
