# Lunar Lander Reinforcement Learning (Dueling Double DQN)

This project implements a **Dueling Double Deep Q-Network (D3QN)** to train an agent to land a lunar lander safely in the OpenAI Gymnasium `LunarLander-v3` environment.

## Project Goal
The objective is to train a reinforcement learning agent that can control the lander's engines (Main, Left, Right) to achieve a soft landing on the landing pad (at coordinates [0,0]). The agent must balance fuel efficiency, stability, and speed while avoiding crashes.

## Installation and Dependencies

### Prerequisites
- Python 3.8+
- [Swig](https://www.swig.org/) (Required for Box2D environment)
    - On macOS: `brew install swig`
    - On Ubuntu: `sudo apt-get install swig`

### Dependencies
The following libraries are required and will be installed via `requirements.txt`:
- `gymnasium[box2d]`: The environment.
- `torch`: Deep learning framework.
- `numpy`: Numerical computations.
- `matplotlib`: Plotting the learning curve.

### Installation
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Estimated Space Requirements
The installation will take approximately **400MB - 600MB** of disk space, primarily due to the PyTorch library (~300MB+ for the macOS/CPU version) and Gymnasium environments.

## Model Architecture
The project uses a **Dueling Double Deep Q-Network (D3QN)**, which features:
- **Dueling Architecture**: Separates the estimation of state Value $V(s)$ and action Advantages $A(s,a)$. This leads to better policy evaluation in complex control tasks where some states are inherently good/bad regardless of the action taken.
- **Double DQN (DDQN)**: Decouples action selection from action evaluation to reduce the overestimation bias common in standard DQN.
- **Soft Target Updates**: Gradually updates the target network parameters to improve training stability.

### Network Structure (`model.py`)
- **Input**: State space (8-dimensional for LunarLander-v3).
- **Shared Feature Layer**: Fully connected layer (128 units, ReLU).
- **Value Stream**: FC layer (128 units, ReLU) -> 1 unit (State Value).
- **Advantage Stream**: FC layer (128 units, ReLU) -> `action_size` units (Action Advantages).

## Hyperparameters and Parameters
Most hyperparameters are defined in `agent.py` and can be adjusted there or via CLI arguments in `train.py`.

| Parameter | Value | Location | Description |
|-----------|-------|----------|-------------|
| `BUFFER_SIZE` | 1e5 | `agent.py` | Experience Replay Buffer capacity |
| `BATCH_SIZE` | 128 | `agent.py` | Training batch size |
| `GAMMA` | 0.99 | `agent.py` | Discount factor for future rewards |
| `TAU` | 1e-3 | `agent.py` | Soft target update interpolation factor |
| `LR` | 1e-3 | `agent.py` | Learning rate for Adam optimizer |
| `UPDATE_EVERY` | 2 | `agent.py` | Update frequency (timesteps) |
| `n_episodes` | 10000 | `train.py` | Default max training episodes |
| `eps_start` | 1.0 | `train.py` | Initial epsilon for exploration |
| `eps_end` | 0.01 | `train.py` | Minimum epsilon |
| `eps_decay` | 0.997| `train.py` | Multiplicative decay factor per episode |

### Reward Shaping (`train.py`)
Custom reward shaping is implemented to accelerate learning:
1. **Fuel Penalty**: Small penalties for firing engines.
2. **Anti-Jitter**: Large penalty for immediate reverse thrust (e.g., Left then Right).
3. **Stability**: Penalty for high vertical acceleration (jerk).
4. **Optimal Path**: Penalty for horizontal deviation from the center line.
5. **Landing Bonus**: Massive bonus (+500) for a successful landing.

## Usage

### 1. Training the Agent
To start training from scratch:
```bash
python train.py --episodes 2000 --out checkpoint.pth
```
Training will automatically stop early if the agent masters the environment (average score >= 280 over 100 episodes).

### 2. Evaluating the Agent (Simulations)
To watch a trained agent play:
```bash
python evaluate.py --checkpoint checkpoint.pth --episodes 5
```

### 3. Clearing Training Data
To reset the project and clear training history:
```bash
rm checkpoint.pth scores.png
```

## Project Structure
- `agent.py`: D3QN Agent implementation.
- `model.py`: Dueling DQN neural network architecture.
- `train.py`: Training loop and reward shaping logic.
- `evaluate.py`: Script to visualize and evaluate trained weights.
- `replay_memory.py`: Experience replay buffer implementation.
- `utils.py`: Plotting and logging utilities.
