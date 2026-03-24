import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Standard Deep Q-Network for 3D Lunar Lander.
    """
    def __init__(self, state_size, action_size, hidden_size1=64, hidden_size2=64):
        """
        Args:
            state_size (int): Dimension of the state space. 
                              (e.g., 14-16 for typical 3D environments)
            action_size (int): Dimension of the action space.
                               (e.g., 6 for typical 3D environments)
            hidden_size1 (int): Number of nodes in the first hidden layer.
            hidden_size2 (int): Number of nodes in the second hidden layer.
        """
        super(DQN, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, action_size)

    def forward(self, state):
        """
        Forward pass converting state tensors -> action values (Q-values).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture.
    Often performs better in complex control tasks like 3D landing because 
    it separates the estimation of state Value and action Advantages.
    """
    def __init__(self, state_size, action_size, hidden_size1=128, hidden_size2=128):
        super(DuelingDQN, self).__init__()
        
        # Feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size1),
            nn.ReLU()
        )
        
        # State Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1)
        )
        
        # Action Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, action_size)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Recombine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
