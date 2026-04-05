import torch
import torch.nn as nn
import torch.nn.functional as F

class PNSPlusMock(nn.Module):
    """
    Mock implementation of PNS+ for local testing.
    The real PNS+ requires a custom C++ CUDA kernel (Normalized Self-attention block)
    which must be compiled in the target environment (e.g., Google Colab).
    This mock allows the training pipeline to run without native extensions.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Basic 3D convolutions to handle temporal data: [Batch, Channels, Time, H, W]
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(32, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        
    def forward(self, x):
        # x is [B, C, T, H, W] expected, though PraNet expects [B, C, H, W]
        # In SUN-SEG literature, PNS+ takes clips of frames.
        
        # If input is 4D [B, C, H, W], unsqueeze to 5D
        if len(x.shape) == 4:
            x = x.unsqueeze(2) 
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        # Squeeze temporal dimension if it's 1 for inference/loss
        if x.shape[2] == 1:
            x = x.squeeze(2)
        elif len(x.shape) == 5:
            # Output the last frame's prediction, or aggregate
            x = x.mean(dim=2) 
            
        return x

# To match standard interface
def get_pns_plus():
    return PNSPlusMock()
