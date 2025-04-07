from torch.nn import Module
from torch import nn
import torch, os
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

class ToyNet(Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.layer1 = nn.Linear(1, 1, bias=False)
        self.layer2 = nn.Linear(1, 1, bias=False)
        self.layer1.weight.data.zero_()
        self.layer1.weight.data += 1
        self.layer2.weight.data.zero_()
        self.layer2.weight.data += 1
    
    def forward(self, x, rank=0):
        # This line is troublesome that by 'rank % 2' that
        # even rank processes will see layer2 used
        # odd rank processes will see layer1 used
        # in conclusion, only one layer is used to compute input x
        layer = f'layer{1 if rank % 2 == 1 else 2}'
        return getattr(self, layer)(x)

def train(rank, world_size, timings_queue):
    setup(rank, world_size)
    # find_unused_parameters will collect unused parameters and it has impacts on gradient
    # if false, will see gradients some of which are None
    # if true, will see gradients all equal
    find_unused_parameters = False
    net = nn.parallel.DistributedDataParallel(ToyNet(), find_unused_parameters=find_unused_parameters)

    # rank 0: tensor([[1.]])
    # rank 1: tensor([[1., 1.]])
    # etc
    input_x = torch.zeros(rank + 1, 1).to(torch.float32) + 1

    # rank 0: output tensor([[1.]])
    #         target tensor([[0.]])
    # rank 1: output tensor([[1., 1.]])
    #         target tensor([[0., 0.]])
    # etc
    output = net(input_x, rank=rank)
    target = torch.zeros_like(input_x)

    # the loss is not normalized, hence higher ranks with larger sample sizes see larger loss values
    # as a result the gradient is proportional to rank
    # Relationship Between Loss and Gradient (pytroch by checking operators used in the loss formula auto computes the gradient):
    # Loss = 0.5 * (rank+1)
    # Gradient = (rank+1)
    # Therefore, gradient = 2 * loss.
    loss = (0.5 * (output - target) ** 2).sum()
    loss.backward()

    print(f'rank {rank}, layer 1, grad {net.module.layer1.weight.grad}')
    print(f'rank {rank}, layer 2, grad {net.module.layer2.weight.grad}')

    dist.destroy_process_group()
    

if __name__=="__main__":

    world_size = 4
    timings_queue = mp.Queue()
    mp.spawn(train, args=(world_size, timings_queue), nprocs=world_size, join=True)

