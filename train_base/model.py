
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from constants import *

class SelfAttention(nn.Module):
    """ Simple Self-Attention Mechanism """
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5  # Scaling factor

    def forward(self, x):
        Q = self.query(x)  # (batch, seq_len, embed_dim)
        K = self.key(x)  # (batch, seq_len, embed_dim)
        V = self.value(x)  # (batch, seq_len, embed_dim)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # Scaled dot-product
        attention_weights = F.softmax(attention_scores, dim=-1)  # Normalize across sequence length
        
        attended_values = torch.matmul(attention_weights, V)  # Weighted sum of values
        return attended_values, attention_weights  # Return both attended features and attention weights

class ResidualUnit(nn.Module):
    def __init__(self, l, w, ar):
        super().__init__()
        self.batchnorm1 = nn.GroupNorm(4, l)
        self.batchnorm2 = nn.GroupNorm(4, l)
        self.batchnorm3 = nn.GroupNorm(4, l)
        self.relu1 = nn.LeakyReLU(0.1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.relu3 = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv1d(l, l, w+2, dilation=ar, padding=(w+1)*ar//2)
        self.conv2 = nn.Conv1d(l, l, w+2, dilation=ar, padding=(w+1)*ar//2)
        self.conv3 = nn.Conv1d(l, l, w+2, dilation=ar, padding=(w+1)*ar//2)

    def forward(self, x, y):
        out = self.conv1(self.relu1(self.batchnorm1(x)))
        out = self.conv2(self.relu2(self.batchnorm2(out)))
        out = self.conv3(self.relu3(self.batchnorm3(out)))
        return x + out, y

class Skip(nn.Module):
    def __init__(self, l):
        super().__init__()
        self.conv = nn.Conv1d(l, l, 1)

    def forward(self, x, y):
        return x, self.conv(x) + y

class thistle(nn.Module):
    def __init__(self, L, W, AR, seq_len=(CL_max+SL)):
        super(thistle, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=15, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=10, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        # Self-Attention Mechanism
        self.attention = SelfAttention(embed_dim=L)

        # Initial skip layer
        self.initial_skip = Skip(L)

        # Residual units
        # self.residual_units = nn.ModuleList()
        # for i, (w, r) in enumerate(zip(W, AR)):
        #     self.residual_units.append(ResidualUnit(L, w, r))
        #     if (i+1) % 4 == 0:
        #         self.residual_units.append(Skip(L))

        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Adaptive instead of fixed

        # Fully connected layers
        self.fc1 = nn.Linear(L, L//2)  # Output from pooled convolutional features # L*2 for LSTM
        self.fc2 = nn.Linear(L//2, 1)  # Output one value

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)  # Drop 30% of neurons randomly

    def forward(self, x):

        # CNN feature extraction
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        #x = self.dropout(F.relu(self.bn2(self.conv2(x))))

        # Add Global Average Pooling (down-sample across the sequence length dimension)
        x = self.global_pool(x)  # Shape: (batch_size, L, 1)

        # Apply self-attention
        # x = x.permute(0, 2, 1)  # Reshape for attention: (batch, seq_len, embed_dim)
        # x, attn_weights = self.attention(x)
        # x = x.permute(0, 2, 1)  # Reshape back: (batch, embed_dim, seq_len)
        # x = self.global_pool(x)  # Apply pooling after attention (this will reduce sequence length)

        # Residual connections
        # x, skip = self.initial_skip(x, 0)
        # for m in self.residual_units:
        #     x, skip = m(x, skip)

        # # Flatten the output for the fully connected layer (batch_size, L)
        x = x.flatten(start_dim=1)  # Flatten the sequence and embedding dimensions into a single vector

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        
        return x
