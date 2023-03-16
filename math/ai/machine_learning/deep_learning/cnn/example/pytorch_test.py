import torch

batch_size = 4
num_anchors = 3
grid_size = 7
bbox_attrs = 5 + 20

x = torch.rand(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

x.select(2, 0).sigmoid_()
x.select(2, 1).sigmoid_()
x.select(2, 4).sigmoid_()

print(x.size())

grid_len = torch.arange(grid_size)
args = torch.meshgrid(grid_len, grid_len, indexing="ij")
x_offset = args[1].contiguous().view(-1, 1)
y_offset = args[0].contiguous().view(-1, 1)


x_y_offset = torch.cat([x_offset, y_offset], 1).repeat([1, num_anchors]).view([-1, 2]).unsqueeze(0)
x[2 : 0 : 2].add_(x_y_offset)


