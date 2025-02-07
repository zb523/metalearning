import os
import torch # type: ignore
import math
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
from pathlib import Path
import time
from torch.amp import autocast, GradScaler # type: ignore
import numpy as np
from torch.utils.data import DataLoader, Dataset # type: ignore
import torch.multiprocessing as mp # type: ignore
from torch.utils.cpp_extension import load_inline # type: ignore
import torch.profiler

# optimize cuda operations
torch.backends.cudnn.benchmark = True
torch.cuda.set_device(0)

# hyperparams for parallel processing
BATCH_SIZE = 32  # can adjust based on memory
N_WORKERS = 4    # for data loading
MIXED_PRECISION = True

# import necessary components from the training file
from python.resaerch.attempt.apd_test_on_transformer import GridworldTransformer, APDAnalysis, vocab, device

vocab_size = len(vocab)

cuda_source = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void fused_softmax_weighted_sum_kernel(
    const scalar_t* __restrict__ raw_weights,  // [batch, n_components]
    const scalar_t* __restrict__ components,     // [batch, n_components, H, W]
    scalar_t* __restrict__ output,               // [batch, H, W]
    const int batch,
    const int n_components,
    const int H,
    const int W) {
  int b = blockIdx.x;
  int idx = threadIdx.x;
  int num_elements = H * W;
  if (b < batch && idx < num_elements) {
      // compute max for numerical stability
      scalar_t max_val = -1e20;
      for (int i = 0; i < n_components; i++) {
          scalar_t w = raw_weights[b * n_components + i];
          if (w > max_val) {
              max_val = w;
          }
      }
      // compute sum of exps
      scalar_t sum_exp = 0;
      for (int i = 0; i < n_components; i++) {
          scalar_t exp_val = exp(raw_weights[b * n_components + i] - max_val);
          sum_exp += exp_val;
      }
      int h = idx / W;
      int w_out = idx % W;
      scalar_t res = 0;
      // compute weighted sum using softmax weights
      for (int i = 0; i < n_components; i++) {
          scalar_t w = raw_weights[b * n_components + i];
          scalar_t softmax_val = exp(w - max_val) / sum_exp;
          // components stored as: [b, i, H, W] flattened in row-major order
          int comp_index = ((b * n_components + i) * H + h) * W + w_out;
          res += softmax_val * components[comp_index];
      }
      output[b * num_elements + idx] = res;
  }
}

at::Tensor fused_softmax_weighted_sum(at::Tensor raw_weights, at::Tensor components) {
  const auto batch = raw_weights.size(0);
  const auto n_components = raw_weights.size(1);
  const auto H = components.size(2);
  const auto W = components.size(3);
  auto output = at::zeros({batch, H, W}, components.options());

  const int num_elements = H * W;
  const int threads = 256;
  // Launch one block per batch element (each block handles H*W threads)
  fused_softmax_weighted_sum_kernel<float><<<batch, threads>>>(
      raw_weights.data_ptr<float>(),
      components.data_ptr<float>(),
      output.data_ptr<float>(),
      batch, n_components, H, W);

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_softmax_weighted_sum", &fused_softmax_weighted_sum, "Fused softmax weighted sum (CUDA)");
}
'''

# compile and load the CUDA extension
# fused_cuda_module = load_inline("fused_cuda_module", cpp_sources="",
#                                  cuda_sources=cuda_source,
#                                  functions=["fused_softmax_weighted_sum"],
#                                  extra_cuda_cflags=["--expt-relaxed-constexpr"])

class SnapshotDataset(Dataset):
    def __init__(self, snapshot_files):
        self.snapshot_files = snapshot_files
        
    def __len__(self):
        return len(self.snapshot_files)
    
    def __getitem__(self, idx):
        snapshot_file = self.snapshot_files[idx]
        epoch = int(snapshot_file.split("_")[-1].split(".")[0])
        return snapshot_file, epoch

def load_model_batch(snapshot_files):
    models = []
    for f in snapshot_files:
        model = GridworldTransformer(vocab_size).to(device, memory_format=torch.channels_last)
        state_dict = torch.load(f, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        models.append(model)
    return models

class ParallelAPDAnalysis(nn.Module):
    def __init__(self, models, n_components=8):
        super().__init__()
        self.n_components = n_components
        self.batch_size = len(models)
        
        # collect params from all models into batched tensors
        self.original_params = []
        self.param_shapes = []
        
        # get first model's shapes
        reference_model = models[0]
        for p in reference_model.parameters():
            if p.ndim == 2:  # only matrix params
                self.param_shapes.append(p.shape)
        
        # stack params from all models
        for shape_idx in range(len(self.param_shapes)):
            stacked_params = torch.stack([
                [p for p in m.parameters() if p.ndim == 2][shape_idx].detach()
                for m in models
            ]).to(device)
            self.original_params.append(stacked_params)

        # initialize batched components and weights
        self.components = nn.ParameterList()
        self.raw_weights = nn.ParameterList()
        
        for shape in self.param_shapes:
            comp = nn.Parameter(
                torch.randn(self.batch_size, n_components, *shape, 
                device=device) / math.sqrt(shape[1])
            )
            w = nn.Parameter(torch.ones(self.batch_size, n_components, device=device))
            self.components.append(comp)
            self.raw_weights.append(w)

        # use native pytorch optimizer with fused operations
        self.apd_optimizer = torch.optim.Adam(
            self.parameters(),
            lr=1e-4,
            fused=True  # uses native cuda fusion when available
        )

        self.scaler = GradScaler()

    @torch.amp.autocast(device_type='cuda')
    def fused_apd_step(self):
        total_loss = 0
        normalized_weights = []
        
        for idx, (orig_p, shape) in enumerate(zip(self.original_params, self.param_shapes)):
            # efficient native pytorch ops
            weights = F.softmax(self.raw_weights[idx], dim=1)
            # use einsum for optimized batched weighted sum
            reconstructed = torch.einsum('bn,bnhw->bhw', weights, self.components[idx])
            loss = F.mse_loss(reconstructed, orig_p)
            total_loss += loss
            normalized_weights.append(weights.detach().cpu().numpy())
            
        return total_loss, normalized_weights

    def update_step(self):
        self.apd_optimizer.zero_grad()
        
        with autocast(device_type='cuda', enabled=MIXED_PRECISION):
            loss, weights = self.fused_apd_step()
            
        if MIXED_PRECISION:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.apd_optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.apd_optimizer.step()
            
        return loss.item(), weights

def run_parallel_apd(models, n_components=8, convergence_threshold=1e-3, max_iterations=1000):
    parallel_apd = ParallelAPDAnalysis(models, n_components).to(device)
    iteration = 0
    apd_loss = float('inf')
    start_time = time.perf_counter()
    print(f"\nStarting APD for batch of {len(models)} models:")
    
    while apd_loss > convergence_threshold and iteration < max_iterations:
        apd_loss, current_weights = parallel_apd.update_step()
        if iteration % 10 == 0:  # log every 10 iterations
            print(f"  Iteration {iteration}: loss = {apd_loss:.6f}")
        iteration += 1
        
    elapsed = time.perf_counter() - start_time
    print(f"  Converged after {iteration} iterations in {elapsed:.2f}s (final loss: {apd_loss:.6f})\n")
    return apd_loss, current_weights, elapsed

def main():
    agent_folder = "agent"
    # sort snapshot files
    snapshot_files = sorted(
        [os.path.join(agent_folder, f) for f in os.listdir(agent_folder) 
         if f.startswith("model_epoch_") and f.endswith(".pt")],
        key=lambda f: int(f.split("_")[-1].split(".")[0])
    )
    
    # create apd_stuff folder
    os.makedirs("apd_stuff", exist_ok=True)
    
    results_log_path = os.path.join("apd_stuff", "apd_results.csv")
    with open(results_log_path, "a") as log:
        log.write("epoch,elapsed_time,apd_loss\n")
    
    # process snapshots in batches
    all_weights = {}
    
    for i in range(0, len(snapshot_files), BATCH_SIZE):
        batch_files = snapshot_files[i:i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{math.ceil(len(snapshot_files)/BATCH_SIZE)}")
        
        # load and process batch of models
        models = load_model_batch(batch_files)
        loss, weights, elapsed = run_parallel_apd(models)
        
        # log results for each model in batch
        for j, f in enumerate(batch_files):
            epoch = int(f.split("_")[-1].split(".")[0])
            print(f"Epoch {epoch}: APD loss {loss:.6f}, time {elapsed:.4f} sec")
            
            with open(results_log_path, "a") as log:
                log.write(f"{epoch},{elapsed:.4f},{loss:.6f}\n")
            
            # collect weights for each layer
            for layer_idx, layer_weights in enumerate(weights):
                key = f"epoch_{epoch}_layer_{layer_idx}"
                all_weights[key] = layer_weights[j]  # get weights for this specific model
    
    # save all weights into one compressed npz file
    np.savez_compressed(os.path.join("apd_stuff", "apd_weights.npz"), **all_weights)
    
    print("APD analysis complete. Results saved in apd_stuff folder.")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profile'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        # run ur model here
        prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == "__main__":
    main() 