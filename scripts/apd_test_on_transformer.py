#!/usr/bin/env python
import os
import math
import random
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# suppress any flash attn warnings
warnings.filterwarnings("ignore", message=".*flash attn.*")

# check for cuda availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device: {device}')

# scale up gridworld parameters with sparse rewards and path penalties
grid_size_param = 20  # bigger grid
optimal_path_length = 2 * (grid_size_param - 1)  # right + down moves to goal

# dynamic start & goal tokens
start_token = "0,0"
goal_token = f"{grid_size_param-1},{grid_size_param-1}"

# expanded vocab to include diagonal moves (will help identify if model learns efficiency)
vocab = {
    "<pad>": 0,
    "<bos>": 1, 
    "<eos>": 2,
    "grid": 3,
    "world": 4,
    "start": 5,
    start_token: 6,
    "goal": 7,
    goal_token: 8,
    "right": 9,
    "down": 10,
    "up": 11,
    "left": 12,
    "diagonal": 13  # new move type to test if model discovers shortcuts
}
inv_vocab = {idx: token for token, idx in vocab.items()}
vocab_size = len(vocab)

# training example with path efficiency pressure
def generate_gridworld(grid_size):
    grid = np.zeros((grid_size, grid_size))
    return grid

def encode_gridworld(grid):
    encoding = ["grid", "world"]
    for row in grid:
        encoding.append(row.tolist())
    return encoding

def calculate_reward(agent_path, grid_size):
    goal_x, goal_y = grid_size - 1, grid_size - 1
    final_x, final_y = agent_path[-1]
    
    if (final_x, final_y) == (goal_x, goal_y):
        reward = 10  # big reward for reaching the goal
        
        # path penalty: discourage inefficient routes
        path_length = len(agent_path) - 1  # exclude start
        efficiency_factor = max(0, 1 - (path_length - optimal_path_length) * 0.1)
        reward += efficiency_factor * 5  # scale efficiency bonus
        
        return reward
    else:
        return -0.1  # small penalty for each step

def generate_episode(grid_size, start_pos=(0, 0), max_steps=200):
    agent_path = [start_pos]
    current_x, current_y = start_pos
    episode = ["<bos>", "start", start_token, "goal", goal_token]
    
    for _ in range(max_steps):
        # basic policy: prefer moving towards the goal
        if current_x < grid_size - 1 and current_y < grid_size - 1:
            move = random.choice(["right", "down", "diagonal"])
        elif current_x < grid_size - 1:
            move = "right"
        elif current_y < grid_size - 1:
            move = "down"
        else:
            break  # agent reached some terminal state
        
        episode.append(move)
        
        # update agent position
        if move == "right":
            current_x += 1
        elif move == "down":
            current_y += 1
        elif move == "diagonal":
            current_x += 1
            current_y += 1
        
        agent_path.append((current_x, current_y))
        
        # check if goal reached
        if (current_x, current_y) == (grid_size - 1, grid_size - 1):
            break
    
    episode.append("<eos>")
    reward = calculate_reward(agent_path, grid_size)
    return episode, reward

def episode_to_indices(episode, vocab):
    return [vocab[token] for token in episode]

# transformer model
class GridworldTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=2,
                 num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        # using batch_first=True to avoid nested tensor warnings (and flash attn issues)
        self.transformer = nn.Transformer(d_model, nhead,
                                          num_encoder_layers,
                                          num_decoder_layers,
                                          dim_feedforward,
                                          batch_first=True)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # src, tgt: shape (batch, seq_len)
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        # pass masks as provided; note: padding masks are not used in this forward pass
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        output = self.fc(output)
        return output

    def generate_square_mask(self, size):
        # generates a causal mask (upper triangular)
        mask = (torch.triu(torch.ones((size, size))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt):
        # For batch_first inputs, src and tgt shape: (batch, seq_len)
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
        tgt_mask = self.generate_square_mask(tgt_seq_len).to(device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device, dtype=torch.bool)
        src_padding_mask = (src == vocab["<pad>"])
        tgt_padding_mask = (tgt == vocab["<pad>"])
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # for batch_first inputs, shape as (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: shape (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# APD analysis (for offline use)
class APDAnalysis(nn.Module):
    def __init__(self, model: nn.Module, n_components: int):
        super().__init__()
        self.original_params = []
        self.param_shapes = []
        for p in model.parameters():
            if p.ndim == 2:  # only analyze weight matrices
                self.original_params.append(p.detach().clone())
                self.param_shapes.append(p.shape)
        self.n_components = n_components

        # register components and weights as parameters so they 
        # can be optimized via torch.optim
        self.components = nn.ParameterList()
        self.raw_weights = nn.ParameterList()  # these will be softmax normalized
        for shape in self.param_shapes:
            comp = nn.Parameter(torch.randn(n_components, *shape) / math.sqrt(shape[1]))
            w = nn.Parameter(torch.ones(n_components))
            self.components.append(comp)
            self.raw_weights.append(w)

        # initialize an optimizer for APD parameters
        self.apd_optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def update_step(self):
        self.apd_optimizer.zero_grad()
        total_loss = 0
        for idx, (orig_p, shape) in enumerate(zip(self.original_params, self.param_shapes)):
            target = orig_p.to(self.raw_weights[idx].device)
            weights = F.softmax(self.raw_weights[idx], dim=0)
            reconstructed = sum(weights[i] * self.components[idx][i] for i in range(self.n_components))
            loss = F.mse_loss(reconstructed, target)
            total_loss += loss
        total_loss_val = total_loss.item()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.apd_optimizer.step()
        normalized_weights = [F.softmax(self.raw_weights[idx], dim=0).detach().cpu().numpy()
                              for idx in range(len(self.raw_weights))]
        return total_loss_val, normalized_weights

    def analyze_final(self, weight_histories):
        print("\nFinal APD Results:")
        for idx, w in enumerate(self.raw_weights):
            final_weights = F.softmax(w, dim=0).detach().cpu().numpy()
            locked_candidate = int(np.argmax(final_weights))
            print(f'Layer {idx}: Final normalized weights: {final_weights}')
            if idx in weight_histories and len(weight_histories[idx]) >= 2:
                candidate_history = [step_weights[locked_candidate] for step_weights in weight_histories[idx]]
                recent = candidate_history[-min(5, len(candidate_history)):]
                std_candidate = np.std(recent)
            else:
                std_candidate = 0.0
            if final_weights[locked_candidate] > 0.8 and std_candidate < 0.02:
                print(f'  -> Candidate locking feature: Component {locked_candidate} (weight = {final_weights[locked_candidate]:.4f}), stable (std = {std_candidate:.4f}).')
            else:
                print(f'  -> No clear locking candidate (max weight = {final_weights[locked_candidate]:.4f}, std = {std_candidate:.4f}).')
        print("APD Analysis complete.\n")

# training loop (apd updates removed; snapshots saved for offline analysis)
def train_model(model, optimizer, vocab, n_epochs=500):
    model.train()
    losses = []
    
    # ensure the agent snapshot folder exists
    os.makedirs("agent", exist_ok=True)
    
    for epoch in range(n_epochs):
        epoch_start_time = time.perf_counter()
        
        episode, reward = generate_episode(grid_size_param)
        indices = episode_to_indices(episode, vocab)
        
        # prepare data for transformer (batch_first: shape (batch, seq_len))
        src = torch.tensor(indices[:-1]).unsqueeze(0).to(device)
        tgt = torch.tensor(indices[1:]).unsqueeze(0).to(device)
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = model.create_mask(src, tgt)
        
        optimizer.zero_grad()
        output = model(src, tgt, src_mask, tgt_mask)
        loss = F.cross_entropy(output.view(-1, vocab_size), tgt.view(-1))
        loss.backward()
        optimizer.step()
        
        agent_epoch_time = (time.perf_counter() - epoch_start_time) * 1000  # ms
        losses.append(loss.item())
        print(f'Agent Epoch {epoch}, Loss: {loss.item():.4f}, Reward: {reward}, Agent training time: {agent_epoch_time:.2f} ms')
        
        # save a snapshot of the model at each epoch for offline APD analysis
        snapshot_path = os.path.join("agent", f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), snapshot_path)
    
    # optionally, save training loss plot
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Agent Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.close()
    
    return model

# main function
def main():
    # initialise model and optimiser
    model = GridworldTransformer(vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("training transformer...")
    model = train_model(model, optimizer, vocab)
    
    print("training complete.")
    print("snapshots saved in 'agent' folder and training loss plot stored in 'training_loss.png'.")

if __name__ == "__main__":
    main()