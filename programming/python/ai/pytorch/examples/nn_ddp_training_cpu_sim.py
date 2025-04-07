import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
process_num = 4
is_used_barrier = False

dataset_size = 2048
batch_size = 512
mini_batch_per_process = batch_size // process_num

# Setup distributed environment
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

# Define a simple neural network with Batch Normalization.
# Note the added barrier after fc1 to synchronize across CPUs.
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        # Compute first linear layer output
        x = self.fc1(x)
        
        # Continue with batch normalization and remaining layers.
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, timings_queue):
    # Set seed for reproducibility (for model initialization etc.)
    torch.manual_seed(SEED)
    timings =[]

    print(f"Starting training on rank {rank}.")
    setup(rank, world_size)
    
    # Create the model and move it to the appropriate device.
    model = SimpleNet()
    device = torch.device("cpu")
    model.to(device)
    
    # Wrap the model in DistributedDataParallel.
    ddp_model = DDP(model)
    
    # Define loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # Create a dummy dataset with static values.
    # All input features are 0.5 and all labels are 0.
    x = torch.randn(dataset_size, 784)  # Reproducible due to fixed seed
    y = torch.randint(0, 10, (dataset_size,))
    dataset = TensorDataset(x, y)
    
    # Create a DistributedSampler and DataLoader.
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=mini_batch_per_process, sampler=sampler)
    
    epochs = 4
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Deterministic shuffling across epochs.
        epoch_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass through the DDP-wrapped model.
            forward_start = time.time()
            outputs = ddp_model(inputs)
            forward_end = time.time()
            timings.append({"rank": rank, "epoch": epoch, "phase": "forward", "start": forward_start, "end": forward_end})

            loss_compute_start = time.time()
            loss = criterion(outputs, labels)
            loss_compute_end = time.time()
            timings.append({"rank": rank, "epoch": epoch, "phase": "loss_compute", "start": loss_compute_start, "end": loss_compute_end})

            # Backward pass: Compute gradients.
            backward_start = time.time()
            loss.backward()
            backward_end = time.time()
            timings.append({"rank": rank, "epoch": epoch, "phase": "backward", "start": backward_start, "end": backward_end})
                        
            # Barrier after loss.backward() to ensure all processes have completed gradient computation.
            sync_start = time.time()
            if dist.is_initialized() and is_used_barrier:
                dist.barrier()
            sync_end = time.time()
            timings.append({"rank": rank, "epoch": epoch, "phase": "sync", "start": sync_start, "end": sync_end})
        
            param_update_start = time.time()
            optimizer.step()
            param_update_end = time.time()
            timings.append({"rank": rank, "epoch": epoch, "phase": "param_update", "start": param_update_start, "end": param_update_end})
        
            epoch_loss += loss.item()

        epoch_sync_start = time.time()
        dist.barrier()
        epoch_sync_end = time.time()
        timings.append({"rank": rank, "epoch": epoch, "phase": "epoch_end_sync", "start": epoch_sync_start, "end": epoch_sync_end})

        # Only the process with rank 0 prints the loss.
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss/len(dataloader):.4f}")

    timings_queue.put(timings)    
    cleanup()


# Plotting function
def plot_timings(timings_df):
    phase_colors = {
        "forward": "cornflowerblue",
        "backward": "peachpuff",
        "overheads": "springgreen",
        "sync": "lightgreen",
        "loss_compute": "orange",
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
    ax.set_title("CPU Simulated NN Training Timeline")
    # Create custom legend entries
    legend_handles = [plt.Rectangle((0,0),1,1, color=c) for c in phase_colors.values()]
    ax.legend(legend_handles, phase_colors.keys(), loc="upper right")
    
    text_str = f"Dataset Size: {dataset_size}\nBatch Size: {batch_size}\nMini-Batch Per Process: {mini_batch_per_process}"
    plt.text(0.84, 0.1, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='left', 
             bbox=dict(boxstyle='round,pad=0.3', edgecolor='grey', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show()

def main():
    world_size = process_num  # Number of processes.
    timings_queue = mp.Queue()
    mp.spawn(train, args=(world_size, timings_queue), nprocs=world_size, join=True)
    all_timings = []
    for _ in range(world_size):
        all_timings.extend(timings_queue.get())

    timings_df = pd.DataFrame(all_timings)
    plot_timings(timings_df)

if __name__ == "__main__":
    main()
