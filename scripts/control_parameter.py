import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from collections import deque
import random
import matplotlib
matplotlib.rcParams['figure.max_open_warning'] = 0  # Suppress max figure warning

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleEnvironment:
    """
    A very simple 1D environment where the agent needs to reach a target position.
    """
    def __init__(self, size=10, max_steps=50):
        self.size = size
        self.max_steps = max_steps
        self.reset()
    
    def reset(self):
        # Agent starts at position 0
        self.position = 0
        self.steps = 0
        # State is just the position normalized to [0,1]
        state = torch.tensor([self.position / self.size], dtype=torch.float32).to(device)
        return state
    
    def step(self, action):
        # Action: 0 = left, 1 = right
        if action == 0 and self.position > 0:
            self.position -= 1
        elif action == 1 and self.position < self.size - 1:
            self.position += 1
        
        self.steps += 1
        
        # Calculate reward: -1 for each step, +10 for reaching the goal
        reward = -1
        done = False
        
        if self.position == self.size - 1:
            reward = 10
            done = True
        elif self.steps >= self.max_steps:
            done = True
        
        # State is just the position normalized to [0,1]
        state = torch.tensor([self.position / self.size], dtype=torch.float32).to(device)
        return state, reward, done

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.to(device)
    
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        # Convert to tensors with proper shapes
        state = torch.cat(state).view(batch_size, -1)
        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_state = torch.cat(next_state).view(batch_size, -1)
        done = torch.tensor(done, dtype=torch.bool).to(device)
        
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class SelectivePressureController:
    """
    Implements the β(s,t) selective pressure function that modulates learning
    based on value estimates and uncertainty.
    
    β(s,t) = β₀ * exp(α * V(s) - γ * U(s)) * C(t)
    
    Where:
    - V(s) is the value estimate of state s
    - U(s) is uncertainty about state s
    - C(t) is a crystallization factor that evolves over time
    - α, γ are control coefficients
    """
    def __init__(self, base_pressure=1.0, value_coef=0.1, uncertainty_coef=0.2, 
                 initial_crystallization=0.5, crystallization_rate=0.01):
        self.base_pressure = base_pressure
        self.value_coef = value_coef
        self.uncertainty_coef = uncertainty_coef
        self.crystallization = initial_crystallization
        self.crystallization_rate = crystallization_rate
        self.state_values = {}
        self.state_uncertainties = {}
        self.visit_counts = {}
        self.pressure_history = []
        self.crystallization_history = []
        
    def discretize_state(self, state):
        """Convert continuous state to discrete key for tracking"""
        return tuple(np.round(state.cpu().numpy().flatten(), 2))
    
    def update_value(self, state, value):
        """Update value estimate for a state"""
        key = self.discretize_state(state)
        if key not in self.state_values:
            self.state_values[key] = value
            self.visit_counts[key] = 1
        else:
            # Running average
            count = self.visit_counts[key]
            self.state_values[key] = (self.state_values[key] * count + value) / (count + 1)
            self.visit_counts[key] += 1
    
    def update_uncertainty(self, state, q_values):
        """Update uncertainty estimate based on q-value variance"""
        key = self.discretize_state(state)
        if q_values.shape[0] > 1:  # If we have multiple actions
            # Use variance between q-values as uncertainty
            uncertainty = torch.var(q_values).item()
        else:
            # Default uncertainty for new states
            uncertainty = 1.0
            
        if key not in self.state_uncertainties:
            self.state_uncertainties[key] = uncertainty
        else:
            # Running average with more weight to recent uncertainty
            self.state_uncertainties[key] = 0.8 * self.state_uncertainties[key] + 0.2 * uncertainty
    
    def update_crystallization(self, performance_improvement):
        """Update crystallization factor based on recent performance"""
        # Increase crystallization as performance improves
        self.crystallization += self.crystallization_rate * performance_improvement
        # Keep within bounds
        self.crystallization = max(0.1, min(2.0, self.crystallization))
        self.crystallization_history.append(self.crystallization)
    
    def calculate_pressure(self, state, q_values):
        """Calculate the selective pressure for a state"""
        key = self.discretize_state(state)
        
        # Get value and uncertainty for the state
        value = self.state_values.get(key, 0.0)
        uncertainty = self.state_uncertainties.get(key, 1.0)
        
        # Update uncertainty estimate
        self.update_uncertainty(state, q_values)
        
        # Calculate pressure using the formula
        pressure = self.base_pressure * np.exp(
            self.value_coef * value - self.uncertainty_coef * uncertainty
        ) * self.crystallization
        
        self.pressure_history.append(pressure)
        return pressure
    
    def reset_history(self):
        """Reset history tracking for a new experiment"""
        self.pressure_history = []
        self.crystallization_history = []

def train_with_selective_pressure(env_size, hidden_dim, learning_rate, gamma, epsilon_start, epsilon_end, 
                                 epsilon_decay, batch_size, buffer_size, episodes, 
                                 pressure_params=None, baseline=False):
    """
    Train an agent with selective pressure control
    
    Args:
        env_size: Size of the environment
        hidden_dim: Hidden dimension of the Q-network
        learning_rate: Learning rate for the optimizer
        gamma: Discount factor
        epsilon_start: Starting epsilon for exploration
        epsilon_end: Final epsilon for exploration
        epsilon_decay: Decay rate for epsilon
        batch_size: Batch size for training
        buffer_size: Size of the replay buffer
        episodes: Number of episodes to train
        pressure_params: Parameters for the selective pressure controller
        baseline: If True, run without selective pressure (for comparison)
    
    Returns:
        Tuple of (rewards_history, steps_history, controller)
    """
    env = SimpleEnvironment(size=env_size)
    state_dim = 1  # Position
    action_dim = 2  # Left or right
    
    # Initialize Q-network and target network
    q_network = QNetwork(state_dim, action_dim, hidden_dim)
    target_network = QNetwork(state_dim, action_dim, hidden_dim)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size)
    
    # Initialize selective pressure controller
    if pressure_params is None:
        pressure_params = {
            'base_pressure': 1.0, 
            'value_coef': 0.1, 
            'uncertainty_coef': 0.2,
            'initial_crystallization': 0.5,
            'crystallization_rate': 0.01
        }
    controller = SelectivePressureController(**pressure_params)
    
    # Metrics to track
    rewards_history = []
    steps_history = []
    avg_rewards = []  # For tracking performance improvement
    
    epsilon = epsilon_start
    
    # Pre-fill buffer with some transitions
    state = env.reset()
    for _ in range(min(buffer_size // 10, 1000)):
        if random.random() < 0.5:
            action = random.randint(0, action_dim - 1)
        else:
            action = 1  # Bias toward right to reach goal occasionally
        
        next_state, reward, done = env.step(action)
        replay_buffer.push(state.view(1, -1), action, reward, next_state.view(1, -1), done)
        
        if done:
            state = env.reset()
        else:
            state = next_state
    
    # Training loop
    for episode in tqdm(range(episodes)):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_values = q_network(state.view(1, -1))
                    action = torch.argmax(q_values).item()
            
            next_state, reward, done = env.step(action)
            episode_reward += reward
            
            # Store transition in replay buffer
            replay_buffer.push(state.view(1, -1), action, reward, next_state.view(1, -1), done)
            
            # Update value estimate in controller
            with torch.no_grad():
                q_values = q_network(state.view(1, -1))
                max_q = torch.max(q_values).item()
            controller.update_value(state, max_q)
            
            state = next_state
            
            # Train if enough samples in buffer
            if len(replay_buffer) > batch_size:
                # Sample a batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
                current_q = q_network(states).gather(1, actions.unsqueeze(1))
                
                # Compute V(s_{t+1}) for all next states
                with torch.no_grad():
                    next_q = target_network(next_states).max(1)[0]
                
                # Compute expected Q values
                expected_q = rewards + gamma * next_q * (~dones)
                
                # Calculate loss with selective pressure
                base_loss = nn.MSELoss(reduction='none')(current_q.squeeze(), expected_q)
                
                # Apply selective pressure if not baseline
                if not baseline:
                    # Calculate pressure for each state in batch
                    pressures = torch.tensor([
                        controller.calculate_pressure(states[i:i+1], q_network(states[i:i+1]))
                        for i in range(batch_size)
                    ], device=device)
                    
                    # Apply pressure to loss
                    weighted_loss = base_loss * pressures
                    loss = weighted_loss.mean()
                else:
                    loss = base_loss.mean()
                
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), 1.0)
                optimizer.step()
        
        # Update target network
        if episode % 10 == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Track metrics
        rewards_history.append(episode_reward)
        steps_history.append(env.steps)
        
        # Calculate moving average of rewards for crystallization update
        if episode >= 10:
            recent_avg = np.mean(rewards_history[-10:])
            if len(avg_rewards) > 0:
                improvement = (recent_avg - avg_rewards[-1]) / abs(avg_rewards[-1] + 1e-8)
            else:
                improvement = 0
            avg_rewards.append(recent_avg)
            
            # Update crystallization based on improvement
            controller.update_crystallization(improvement)
    
    return rewards_history, steps_history, controller

def compare_methods(env_size=20, hidden_dim=32, episodes=300):
    """
    Compare training with and without selective pressure.
    
    Args:
        env_size: Size of the environment
        hidden_dim: Hidden dimension of the Q-network
        episodes: Number of episodes to train
    """
    # Common parameters
    learning_rate = 0.001
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.99
    batch_size = 64
    buffer_size = 10000
    
    # Different selective pressure configurations
    pressure_configs = [
        # Baseline (no pressure)
        {"name": "Baseline", "params": None, "baseline": True},
        
        # Standard selective pressure
        {"name": "Selective Pressure", "params": {
            'base_pressure': 1.0, 
            'value_coef': 0.1, 
            'uncertainty_coef': 0.2,
            'initial_crystallization': 0.5,
            'crystallization_rate': 0.01
        }, "baseline": False},
        
        # Higher value emphasis
        {"name": "Value Focus", "params": {
            'base_pressure': 1.0, 
            'value_coef': 0.3,  # Higher emphasis on value
            'uncertainty_coef': 0.1,  # Lower penalty for uncertainty
            'initial_crystallization': 0.5,
            'crystallization_rate': 0.01
        }, "baseline": False},
        
        # Higher uncertainty emphasis (more exploration)
        {"name": "Uncertainty Focus", "params": {
            'base_pressure': 1.0, 
            'value_coef': 0.1,
            'uncertainty_coef': 0.4,  # Higher penalty for uncertainty
            'initial_crystallization': 0.5,
            'crystallization_rate': 0.02  # Faster crystallization
        }, "baseline": False}
    ]
    
    results = {}
    
    # Run each configuration
    for config in pressure_configs:
        print(f"\nTraining with {config['name']} configuration...")
        rewards, steps, controller = train_with_selective_pressure(
            env_size=env_size,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size,
            buffer_size=buffer_size,
            episodes=episodes,
            pressure_params=config['params'],
            baseline=config['baseline']
        )
        
        results[config['name']] = {
            'rewards': rewards,
            'steps': steps
        }
        
        # Add controller data if available
        if not config['baseline']:
            results[config['name']]['pressure'] = controller.pressure_history
            results[config['name']]['crystallization'] = controller.crystallization_history
    
    # Create output directory
    os.makedirs("images", exist_ok=True)
    
    # Plot results
    plot_comparison(results, episodes)

def plot_comparison(results, episodes):
    """
    Plot comparison of different methods.
    
    Args:
        results: Dictionary of results
        episodes: Number of episodes
    """
    # Create a figure with two subplots (rewards and selective pressure)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot rewards
    for name, data in results.items():
        rewards = data['rewards']
        # Smooth rewards for better visualization
        window_size = 10
        smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, episodes), smoothed_rewards, label=name)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Learning Performance Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot selective pressure for non-baseline methods
    for name, data in results.items():
        if 'pressure' in data:
            pressure = data['pressure']
            ax2.plot(pressure, label=f"{name} Pressure")
            
            # Mark when pressure changes significantly
            pressure_array = np.array(pressure)
            # Calculate rate of change
            diffs = np.abs(np.diff(pressure_array))
            # Find significant changes (top 5%)
            threshold = np.percentile(diffs, 95)
            significant_changes = np.where(diffs > threshold)[0]
            
            # Mark significant changes as potential phase transitions
            for idx in significant_changes:
                ax2.axvline(x=idx, color='gray', linestyle='--', alpha=0.2)
    
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Selective Pressure')
    ax2.set_title('Selective Pressure Dynamics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("images/control_parameter_analysis.png")
    plt.close()
    
    # Plot crystallization factor separately
    plt.figure(figsize=(10, 5))
    for name, data in results.items():
        if 'crystallization' in data:
            crystallization = data['crystallization']
            plt.plot(crystallization, label=f"{name} Crystallization")
    
    plt.xlabel('Episode')
    plt.ylabel('Crystallization Factor')
    plt.title('Evolution of the Crystallization Control Parameter')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("images/crystallization_evolution.png")
    plt.close()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True
    
    # Run comparison with a larger environment to better show effects
    start_time = time.time()
    compare_methods(env_size=30, hidden_dim=32, episodes=300)
    end_time = time.time()
    
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds") 