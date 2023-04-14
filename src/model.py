import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNTransformerClassifier(nn.Module):
    def __init__(self, num_classes, img_channels=3, embed_dim=32, num_heads=4, hidden_dim=64):
        super(CNNTransformerClassifier, self).__init__()
        self.conv1 = nn.Conv2d(img_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.transformer = nn.Transformer(d_model=embed_dim, nhead=num_heads)
        
        self.fc = nn.Linear(embed_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Flatten the feature maps
        B, C, H, W = x.size()
        x = x.view(B, C, -1).transpose(1, 2)
        
        # Apply the transformer layer
        x = self.transformer(x)
        
        # Take the mean across the sequence dimension
        x = x.mean(dim=1)
        
        x = F.relu(self.fc(x))
        x = self.out(x)
        
        return x
