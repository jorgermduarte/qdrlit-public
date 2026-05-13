import torch
import torch.nn as nn

class ExtraLayer(nn.Module):
    """
    Extra layer component for the QDQN network.
    Implements a simple linear projection with residual connection.
    """
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        # Linear projection layer for residual connection
        self.residual_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Apply linear transformation for residual connection
        residual = self.residual_proj(x)
        # Debug print to monitor residual norm during training
        print(f"q_out residual norm: {torch.norm(residual)}")
        return residual

class QDQN(nn.Module):
    """
    Deep Q-Network (DQN) architecture for Q-value approximation.
    
    Architecture:
    - Input layer: state_size → 256
    - Layer normalization for training stability
    - Extra layer with residual connection
    - Hidden layer: 256 → 128
    - Output layer: 128 → action_size (Q-values for all actions)
    """
    def __init__(self, input_size=45, output_size=90):
        super().__init__()
        # Sequential network architecture
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),      # Input to hidden layer
            nn.LayerNorm(256),               # Layer normalization for stability
            ExtraLayer(input_dim=256, output_dim=256),  # Residual connection layer
            nn.ReLU(),                       # Non-linear activation
            nn.Linear(256, 128),             # Hidden to hidden layer
            nn.ReLU(),                       # Non-linear activation
            nn.Linear(128, output_size)      # Hidden to output layer (Q-values)
        )

    def forward(self, x):
        # Ensure input is a PyTorch tensor
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        # Forward pass through the network
        output = self.net(x)
        
        # Handle single sample case: remove batch dimension if present
        if output.dim() == 2 and output.size(0) == 1:
            output = output.squeeze(0)
        
        return output
