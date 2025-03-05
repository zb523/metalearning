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
        state = torch.tensor([self.position / self.size], dtype=torch.float32).to(device)  # tensor shape: [1]
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
        state = torch.tensor([self.position / self.size], dtype=torch.float32).to(device)  # tensor shape: [1]
        return state, reward, done

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # input: [batch, state_dim], output: [batch, hidden_dim]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # input: [batch, hidden_dim], output: [batch, hidden_dim]
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)   # input: [batch, hidden_dim], output: [batch, action_dim]
        )
        self.to(device)
    
    def forward(self, x):
        # x shape: [batch, state_dim]
        # output shape: [batch, action_dim]
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
        state = torch.cat(state).view(batch_size, -1)  # tensor shape: [batch_size, state_dim]
        action = torch.tensor(action, dtype=torch.long).to(device)  # tensor shape: [batch_size]
        reward = torch.tensor(reward, dtype=torch.float32).to(device)  # tensor shape: [batch_size]
        next_state = torch.cat(next_state).view(batch_size, -1)  # tensor shape: [batch_size, state_dim]
        done = torch.tensor(done, dtype=torch.bool).to(device)  # tensor shape: [batch_size]
        
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

def calculate_order_parameters(model, env, num_samples=1000):
    """
    Calculate potential order parameters for the current model
    """
    # Sample states uniformly across the environment
    states = torch.linspace(0, 1, num_samples).view(-1, 1).to(device)  # tensor shape: [num_samples, 1]
    
    # Get Q-values for all states
    with torch.no_grad():
        q_values = model(states)  # tensor shape: [num_samples, action_dim]
    
    # Calculate various potential order parameters
    
    # 1. Value difference (max Q - min Q) - measures the agent's ability to discriminate
    value_range = torch.max(q_values) - torch.min(q_values)
    
    # 2. Action certainty - average difference between best and second best action
    sorted_q, _ = torch.sort(q_values, dim=1, descending=True)  # tensor shape: [num_samples, action_dim]
    if sorted_q.shape[1] > 1:  # If we have more than one action
        action_certainty = torch.mean(sorted_q[:, 0] - sorted_q[:, 1])
    else:
        action_certainty = torch.tensor(0.0).to(device)
    
    # 3. Value entropy - measures the spread of Q-values
    q_softmax = torch.softmax(q_values, dim=1)  # tensor shape: [num_samples, action_dim]
    entropy = -torch.sum(q_softmax * torch.log(q_softmax + 1e-10), dim=1).mean()  # tensor shape: scalar
    
    # 4. Gradient magnitude - requires gradient calculation
    states.requires_grad_(True)
    q_values = model(states)  # tensor shape: [num_samples, action_dim]
    q_sum = torch.sum(q_values)  # tensor shape: scalar
    q_sum.backward()
    gradient_magnitude = torch.norm(states.grad).item()  # tensor shape: scalar
    
    return {
        "value_range": value_range.item(),
        "action_certainty": action_certainty.item(),
        "entropy": entropy.item(),
        "gradient_magnitude": gradient_magnitude
    }

def train_agent(env_size, hidden_dim, learning_rate, gamma, epsilon_start, epsilon_end, 
                epsilon_decay, batch_size, buffer_size, episodes, target_update=10):
    """
    Train an agent and track metrics to identify order parameters
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
    
    # Metrics to track
    rewards_history = []
    steps_history = []
    order_parameters_history = []
    
    epsilon = epsilon_start
    
    # Collect multiple transitions before training to speed up initial phase
    state = env.reset()
    for _ in range(min(buffer_size // 10, 1000)):  # Pre-fill buffer with some transitions
        if random.random() < 0.5:  # Random actions for initial exploration
            action = random.randint(0, action_dim - 1)
        else:
            action = 1  # Bias toward right to reach goal occasionally
        
        next_state, reward, done = env.step(action)
        replay_buffer.push(state.view(1, -1), action, reward, next_state.view(1, -1), done)
        
        if done:
            state = env.reset()
        else:
            state = next_state
    
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
                    q_values = q_network(state.view(1, -1))  # tensor shape: [1, action_dim]
                    action = torch.argmax(q_values).item()
            
            next_state, reward, done = env.step(action)
            episode_reward += reward
            
            # Store transition in replay buffer
            replay_buffer.push(state.view(1, -1), action, reward, next_state.view(1, -1), done)
            state = next_state
            
            # Train if enough samples in buffer - do multiple batches at once for efficiency
            if len(replay_buffer) > batch_size:
                # Train on multiple batches to speed up learning
                num_batches = 4  # Increase batch processing
                for _ in range(num_batches):
                    # Sample a batch
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                    
                    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
                    current_q = q_network(states).gather(1, actions.unsqueeze(1))  # tensor shape: [batch_size, 1]
                    
                    # Compute max Q(s_{t+1}, a) for all next states
                    with torch.no_grad():
                        next_q = target_network(next_states).max(1)[0]  # tensor shape: [batch_size]
                    
                    # Compute the expected Q values
                    expected_q = rewards + gamma * next_q * (~dones)  # tensor shape: [batch_size]
                    
                    # Compute loss
                    loss = nn.MSELoss()(current_q.squeeze(), expected_q)  # tensor shape: scalar
                    
                    # Optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        # Update target network
        if episode % target_update == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Calculate order parameters less frequently to save computation
        if episode % 20 == 0:  # Calculate every 20 episodes instead of 10
            order_params = calculate_order_parameters(q_network, env)
            order_parameters_history.append((episode, order_params))
        
        rewards_history.append(episode_reward)
        steps_history.append(env.steps)
    
    return rewards_history, steps_history, order_parameters_history, q_network

def run_parameter_sweep():
    """
    Run a parameter sweep to identify phase transitions
    """
    # Reduced parameter sweep for faster execution
    env_sizes = [5, 15, 30]  # Reduced from 4 to 3 values with wider spread
    hidden_dims = [8, 32, 64]  # Reduced from 4 to 3 values
    learning_rates = [0.01]  # Fixed learning rate to reduce combinations
    
    # Fixed parameters - reduced episodes for faster execution
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.99  # Faster decay
    batch_size = 128  # Increased batch size for faster learning
    buffer_size = 10000
    episodes = 300  # Reduced from 500
    
    all_results = []
    
    # Create a figure for representative examples
    fig_examples, axes_examples = plt.subplots(3, 3, figsize=(15, 15))
    fig_examples.suptitle("Representative Learning Curves", fontsize=16)
    
    # Track data for consolidated plots
    complexity_ratios = []  # env_size / hidden_dim
    performance_values = []
    order_param_values = {
        "value_range": [],
        "action_certainty": [],
        "entropy": [],
        "gradient_magnitude": []
    }
    
    for i, env_size in enumerate(env_sizes):
        for j, hidden_dim in enumerate(hidden_dims):
            for learning_rate in learning_rates:
                print(f"\nTraining with env_size={env_size}, hidden_dim={hidden_dim}, learning_rate={learning_rate}")
                
                rewards, steps, order_params, model = train_agent(
                    env_size=env_size,
                    hidden_dim=hidden_dim,
                    learning_rate=learning_rate,
                    gamma=gamma,
                    epsilon_start=epsilon_start,
                    epsilon_end=epsilon_end,
                    epsilon_decay=epsilon_decay,
                    batch_size=batch_size,
                    buffer_size=buffer_size,
                    episodes=episodes
                )
                
                # Store results for analysis
                final_reward = np.mean(rewards[-10:])
                all_results.append({
                    "env_size": env_size,
                    "hidden_dim": hidden_dim,
                    "learning_rate": learning_rate,
                    "final_reward": final_reward,
                    "final_steps": np.mean(steps[-10:]),
                    "order_params": order_params[-1][1]  # Last recorded order parameters
                })
                
                # Plot representative examples in the grid
                if i < 3 and j < 3:  # Only plot the first 3x3 grid
                    ax = axes_examples[i, j]
                    ax.plot(rewards)
                    ax.set_title(f"Size={env_size}, Hidden={hidden_dim}")
                    ax.set_xlabel("Episode")
                    ax.set_ylabel("Reward")
                
                # Collect data for consolidated plots
                complexity_ratio = env_size / hidden_dim
                complexity_ratios.append(complexity_ratio)
                performance_values.append(final_reward)
                
                for param_name, param_value in order_params[-1][1].items():
                    order_param_values[param_name].append(param_value)
    
    # Save the representative examples figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("images/representative_examples.png")
    plt.close()
    
    # Analyze results to identify the order parameter
    analyze_order_parameters(all_results, complexity_ratios, performance_values, order_param_values)

def analyze_order_parameters(results, complexity_ratios, performance_values, order_param_values):
    """
    Analyze the results to identify which order parameter best predicts phase transitions
    """
    # Create output directory if it doesn't exist
    os.makedirs("images", exist_ok=True)
    
    # Get all order parameter names
    param_names = list(results[0]["order_params"].keys())
    
    # Calculate correlation between each order parameter and performance
    correlations = {}
    for param_name in param_names:
        param_values = [r["order_params"][param_name] for r in results]
        correlation = np.corrcoef(param_values, [r["final_reward"] for r in results])[0, 1]
        correlations[param_name] = correlation
    
    # Print results
    print("\n===== ORDER PARAMETER ANALYSIS =====")
    print("Correlation with final performance:")
    for param_name, correlation in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"{param_name}: {correlation:.4f}")
    
    # Identify the best order parameter
    best_param = max(correlations.items(), key=lambda x: abs(x[1]))[0]
    print(f"\nThe best order parameter appears to be: {best_param}")
    
    # Create consolidated plots
    create_consolidated_plots(results, best_param, complexity_ratios, performance_values, order_param_values, correlations)

def create_consolidated_plots(results, best_param, complexity_ratios, performance_values, order_param_values, correlations):
    """
    Create consolidated plots that show the key relationships
    """
    # 1. Create a phase diagram showing environment size vs hidden dimension
    env_sizes = sorted(list(set(r["env_size"] for r in results)))
    hidden_dims = sorted(list(set(r["hidden_dim"] for r in results)))
    
    # Create a grid for the phase diagram
    phase_grid = np.zeros((len(env_sizes), len(hidden_dims)))
    performance_grid = np.zeros((len(env_sizes), len(hidden_dims)))
    
    # Fill the grid with the best order parameter values
    for r in results:
        i = env_sizes.index(r["env_size"])
        j = hidden_dims.index(r["hidden_dim"])
        phase_grid[i, j] = r["order_params"][best_param]
        performance_grid[i, j] = r["final_reward"]
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot the phase diagram
    im1 = ax1.imshow(phase_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(im1, ax=ax1, label=best_param)
    ax1.set_xticks(np.arange(len(hidden_dims)))
    ax1.set_yticks(np.arange(len(env_sizes)))
    ax1.set_xticklabels(hidden_dims)
    ax1.set_yticklabels(env_sizes)
    ax1.set_xlabel("Hidden Dimension")
    ax1.set_ylabel("Environment Size")
    ax1.set_title(f"Phase Diagram: {best_param}")
    
    # Add contour lines for performance thresholds
    contour_levels = np.linspace(np.min(performance_grid), np.max(performance_grid), 5)
    CS = ax1.contour(performance_grid, levels=contour_levels, colors='white', alpha=0.6)
    ax1.clabel(CS, inline=True, fontsize=8, fmt='%.1f')
    
    # Plot the performance diagram
    im2 = ax2.imshow(performance_grid, cmap='plasma', interpolation='nearest')
    plt.colorbar(im2, ax=ax2, label="Final Reward")
    ax2.set_xticks(np.arange(len(hidden_dims)))
    ax2.set_yticks(np.arange(len(env_sizes)))
    ax2.set_xticklabels(hidden_dims)
    ax2.set_yticklabels(env_sizes)
    ax2.set_xlabel("Hidden Dimension")
    ax2.set_ylabel("Environment Size")
    ax2.set_title("Performance Diagram")
    
    plt.tight_layout()
    plt.savefig("images/consolidated_phase_diagram.png")
    plt.close()
    
    # 2. Create a transition plot showing how performance and order parameters vary with complexity
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort all data by complexity ratio
    sorted_indices = np.argsort(complexity_ratios)
    sorted_complexity = np.array(complexity_ratios)[sorted_indices]
    sorted_performance = np.array(performance_values)[sorted_indices]
    
    # Normalize performance for plotting
    norm_performance = (sorted_performance - np.min(sorted_performance)) / (np.max(sorted_performance) - np.min(sorted_performance) + 1e-10)
    
    # Plot performance
    ax.plot(sorted_complexity, norm_performance, 'k-', linewidth=2, label="Performance")
    
    # Plot top 3 order parameters
    top_params = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    colors = ['r', 'g', 'b']
    
    for i, (param_name, _) in enumerate(top_params):
        param_values = np.array(order_param_values[param_name])[sorted_indices]
        # Normalize parameter values
        norm_values = (param_values - np.min(param_values)) / (np.max(param_values) - np.min(param_values) + 1e-10)
        ax.plot(sorted_complexity, norm_values, f'{colors[i]}-', linewidth=2, label=param_name)
    
    # Add vertical lines at potential phase transitions
    # Find points where performance changes rapidly
    perf_diff = np.abs(np.diff(norm_performance))
    threshold = np.percentile(perf_diff, 90)  # Top 10% of changes
    transition_points = np.where(perf_diff > threshold)[0]
    
    for point in transition_points:
        ax.axvline(x=sorted_complexity[point], color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel("Complexity Ratio (Environment Size / Hidden Dimension)")
    ax.set_ylabel("Normalized Value")
    ax.set_title("Phase Transitions in Learning Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("images/transition_analysis.png")
    plt.close()
    
    # Print the phase diagram as text
    print("\n===== PHASE DIAGRAM =====")
    print(f"Order Parameter: {best_param}")
    print("\nEnvironment Size × Hidden Dimension")
    header = "Env\\Hidden " + " ".join([f"{h:8d}" for h in hidden_dims])
    print(header)
    for i, env_size in enumerate(env_sizes):
        row = f"{env_size:11d} " + " ".join([f"{phase_grid[i, j]:8.4f}" for j in range(len(hidden_dims))])
        print(row)
    
    print("\n===== PERFORMANCE DIAGRAM =====")
    print("\nEnvironment Size × Hidden Dimension")
    print(header)
    for i, env_size in enumerate(env_sizes):
        row = f"{env_size:11d} " + " ".join([f"{performance_grid[i, j]:8.4f}" for j in range(len(hidden_dims))])
        print(row)

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        # Enable cuDNN benchmark for faster training
        torch.backends.cudnn.benchmark = True
    
    start_time = time.time()
    run_parameter_sweep()
    end_time = time.time()
    
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds") 