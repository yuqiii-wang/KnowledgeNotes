import torch
import torch.profiler as profiler
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

emb_dim=256
num_heads=4
num_experts = 16

# Setup distributed environment
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# Attention Module
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim) # 3 for q, k, v
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, rank, timings, epoch):
        attn_start = time.time()
        B, N, C = x.shape
        
        # Generate q, k, v and split into heads
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shapes: (B, num_heads, N, head_dim)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)  # (B, num_heads, N, head_dim)
        
        # Combine heads and project
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N, self.embed_dim)
        out = self.fc_out(attn_output)

        attn_end = time.time()
        timings.append({"rank": rank, "epoch": epoch, "phase": "attention", "start": attn_start, "end": attn_end})
        return out


# MoE Transformer Layer with Attention
class MoETransformerLayer(nn.Module):
    def __init__(self, layer_id, rank, world_size):
        super().__init__()
        self.layer_id = layer_id
        self.attention = SelfAttention(emb_dim, num_heads)
        self.experts = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(emb_dim, num_experts)
        # Add layer normalization for attention and MoE outputs
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x, rank, timings, epoch):
        # Attention block with residual connection and layer norm
        start = time.time()
        attn_output = self.attention(x, rank, timings, epoch)
        end = time.time()
        timings.append({"rank": rank, "epoch": epoch, "phase": "attention", "start": start, "end": end})
        x = x + attn_output  # Residual connection
        x = self.norm1(x)

        # MoE block with residual connection and layer norm
        residual = x  # Save for residual connection
        gate_scores = self.gate(x)
        gate_weights = torch.softmax(gate_scores, dim=-1)
        
        expert_outputs = [expert(x) for expert in self.experts]
        stacked_experts = torch.stack(expert_outputs, dim=-1)
        
        moe_output = (stacked_experts * gate_weights.unsqueeze(-2)).sum(dim=-1)
        x = residual + moe_output  # Apply residual connection
        x = self.norm2(x)  # Layer normalization after MoE
        end = time.time()
        timings.append({"rank": rank, "epoch": epoch, "phase": "moe", "start": timings[-1]["end"], "end": end})
   
        return x

# MoE Model
class MoEModel(nn.Module):
    def __init__(self, rank, world_size):
        super().__init__()
        self.layer1 = MoETransformerLayer(1, rank, world_size)
        self.layer2 = MoETransformerLayer(2, rank, world_size)
        self.layer3 = MoETransformerLayer(3, rank, world_size)

    def forward(self, x, rank, timings, epoch):
        x = self.layer1(x, rank, timings, epoch)
        sync1_end = time.time()
        timings.append({"rank": rank, "epoch": epoch, "phase": "overheads", "start": timings[-1]["end"], "end": sync1_end})

        x = self.layer2(x, rank, timings, epoch)
        sync2_end = time.time()
        timings.append({"rank": rank, "epoch": epoch, "phase": "overheads", "start": timings[-1]["end"], "end": sync2_end})

        x = self.layer3(x, rank, timings, epoch)
        sync3_end = time.time()
        timings.append({"rank": rank, "epoch": epoch, "phase": "overheads", "start": timings[-1]["end"], "end": sync3_end})

        return x

# Training function
def train(rank, world_size, timings_queue):
    setup(rank, world_size)
    model = MoEModel(rank, world_size)
    ddp_model = nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    data = torch.randn(256, emb_dim).unsqueeze(1)  # Add sequence dimension
    targets = torch.randn(256, emb_dim)
    timings = []

    for epoch in range(3):
        optimizer.zero_grad()
        outputs = ddp_model(data, rank, timings, epoch)
        
        loss_start = time.time()
        loss = loss_fn(outputs.squeeze(1), targets)
        loss_end = time.time()
        timings.append({"rank": rank, "epoch": epoch, "phase": "loss", "start": loss_start, "end": loss_end})

        backward_start = time.time()
        loss.backward()
        backward_end = time.time()
        timings.append({"rank": rank, "epoch": epoch, "phase": "backward", "start": backward_start, "end": backward_end})

        param_update_start = time.time()
        optimizer.step()
        param_update_end = time.time()
        timings.append({"rank": rank, "epoch": epoch, "phase": "param_update", "start": param_update_start, "end": param_update_end})

        epoch_sync_start = time.time()
        dist.barrier()
        epoch_sync_end = time.time()
        timings.append({"rank": rank, "epoch": epoch, "phase": "epoch_end_sync", "start": epoch_sync_start, "end": epoch_sync_end})

    timings_queue.put(timings)
    dist.destroy_process_group()


# Plotting function
def plot_timings(timings_df):
    phase_colors = {
        "attention": "royalblue",
        "moe": "cornflowerblue",
        "overheads": "lightgreen",
        "loss": "orange",
        "backward": "peachpuff",
        "param_update": "skyblue",
        "epoch_end_sync": "deepskyblue"
    }

    global_start = timings_df["start"].min()
    timings_df["start"] -= global_start
    timings_df["end"] -= global_start
    timings_df["duration"] = timings_df["end"] - timings_df["start"]

    fig, ax = plt.subplots(figsize=(15, 6))
    # Plot each timing as a thin horizontal bar (height=0.4)
    for _, row in timings_df.iterrows():
        ax.barh(f"Device {row['rank']}", row["duration"], left=row["start"],
                color=phase_colors[row["phase"]], height=0.4)

    # For each epoch, add a vertical dashed line at             the epoch end synchronization time.
    # We pick the first occurrence of an "epoch_end_sync" event per epoch.
    epochs = sorted(timings_df["epoch"].unique())
    for epoch in epochs:
        mask = (timings_df["phase"] == "epoch_end_sync") & (timings_df["epoch"] == epoch)
        if mask.any():
            epoch_end_time = timings_df[mask].iloc[0]["end"]
            ax.axvline(x=epoch_end_time, color='lightgrey', linestyle='--', linewidth=1)
            # Add text beside the dashed line. Adjust y coordinate as needed.
            ax.text(epoch_end_time, 0.5, f"{epoch+1}-th epoch ends", color='grey', rotation=90,
                    verticalalignment='bottom', horizontalalignment='right')

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Device (CPU)")
    ax.set_title("CPU Simulated MoE Transformer Training Timeline")
    # Create custom legend entries
    legend_handles = [plt.Rectangle((0,0),1,1, color=c) for c in phase_colors.values()]
    ax.legend(legend_handles, phase_colors.keys(), loc="upper right")
    plt.tight_layout()
    plt.show()


# Main execution
def main():
    world_size = 4
    timings_queue = mp.Queue()
    mp.spawn(train, args=(world_size, timings_queue), nprocs=world_size, join=True)

    all_timings = []
    for _ in range(world_size):
        all_timings.extend(timings_queue.get())

    timings_df = pd.DataFrame(all_timings)
    plot_timings(timings_df)


if __name__ == "__main__":
    main()
