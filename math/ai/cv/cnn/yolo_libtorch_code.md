# YOLOv3 + Libtorch Implementation

CUDA/CPU switch is set in the code below.
```cpp
torch::DeviceType device_type;

if (torch::cuda::is_available() ) {        
    device_type = torch::kCUDA;
} else {
    device_type = torch::kCPU;
}
torch::Device device(device_type);

// input image size for YOLO v3
int input_image_size = 416;

Darknet net("../models/yolov3.cfg", &device);
```

### Prerequisites

A typical `torch::Tensor x` would have 4 dimensions:
* `x.sizes[0]` for batch size, the number of images contained in one batch
* `x.sizes[1]` for image channels, typically $3$ for RGB
* `x.sizes[2]` for image width
* `x.sizes[3]` for image height

After having finished config file read, run each layer stacking to its previous forming a input/output sequential ; there is `module_count = 53` .

```cpp
torch::Tensor Darknet::forward(torch::Tensor x) 
{
    for (size_t i = 0; i < module_count; i++)
    {
        std::map<string, string> block = blocks[i+1];
		std::string layer_type = block["type"];

        if (layer_type == "convolutional" || layer_type == "upsample" || layer_type == "maxpool")
		{
			torch::nn::SequentialImpl *seq_imp = dynamic_cast<torch::nn::SequentialImpl *>(module_list[i].ptr().get());
			
			x = seq_imp->forward(x);
			outputs[i] = x;
		}
		else if (layer_type == "route")
        {
			int start = std::stoi(block["start"]);
			int end = std::stoi(block["end"]);

    		torch::Tensor map_1 = outputs[i + start];
    		torch::Tensor map_2 = outputs[i + end];

    		x = torch::cat({map_1, map_2}, 1);
        }
        else if (layer_type == "shortcut")
		{
			int from = std::stoi(block["from"]);
			x = outputs[i-1] + outputs[i+from];
            outputs[i] = x;
		}
		else if (layer_type == "yolo")
		{
			torch::nn::SequentialImpl *seq_imp = dynamic_cast<torch::nn::SequentialImpl *>(module_list[i].ptr().get());
			x = seq_imp->forward(x, inp_dim, num_classes, *_device);

            outputs[i] = outputs[i-1];
        }
    }
}
``` 

## Upsample, Convolutional and MaxPooling Layer

For convolutional layer, `libtorch` has built-in structure such as 
```cpp
torch::nn::Conv2d conv = torch::nn::Conv2d(conv_options(prev_filters, filters, kernel_size, stride, pad, 1, with_bias));
module->push_back(conv);
```

Upsample is same as interpolation; by default, it is a stepped interpolation such as
```
tensor([[[[1., 2.],
          [3., 4.]]]])
->
tensor([[[[1., 1., 2., 2.],
          [1., 1., 2., 2.],
          [3., 3., 4., 4.],
          [3., 3., 4., 4.]]]])
```


```cpp
struct UpsampleLayer : torch::nn::Module
{
    ...
    torch::Tensor forward(torch::Tensor x) {

  		w = sizes[2] * _stride;
  		h = sizes[3] * _stride;

    	x = torch::upsample_nearest2d(x, {w, h});

        return x;
    }
};
```

MaxPooling select the max value from given kernel size at a stride `MaxPoolLayer2D(int kernel_size, int stride)`

```cpp
struct MaxPoolLayer2D : torch::nn::Module
{
    ...
    torch::Tensor forward(torch::Tensor x) {	

 		x = torch::max_pool2d(x, {_kernel_size, _kernel_size}, {_stride, _stride});    

        return x;
    }
};
```

## Route and Shortcut Layer

Route simply concatenates two layers' outputs into one
```cpp
torch::Tensor map_1 = outputs[i + start];
torch::Tensor map_2 = outputs[i + end];

x = torch::cat({map_1, map_2}, 1);
``` 

where `torch.cat(...)` performs the concatenation operation.
```
>>> x = torch.randn(2, 3)
tensor([[ 0.6580, -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497]])
>>> torch.cat((x, x, x), 1)
tensor([[ 0.6580, -1.0969, -0.4614,  0.6580, -1.0969, -0.4614,  0.6580,
         -1.0969, -0.4614],
        [-0.1034, -0.5790,  0.1497, -0.1034, -0.5790,  0.1497, -0.1034,
         -0.5790,  0.1497]])
```

Shortcut does the sum out of two layers
```cpp
x = outputs[i-1] + outputs[i+from];
```

## Detection Layer

Decode from the final dense layer with the output encoded in tensors of the size $S \times S \times (5 \times B + C)$.

Read `[yolo]` layer from config file then 
* $9$ anchors scattered at different image coordinates
* mask is used to set what anchors are used, for example, `mask = 3,4,5` indicates `anchors = 30,61,  62,45,  59,119` are considered only
* `jitter` can be $\in [0,1]$ and used to crop images during training for data augumentation. 

```
[yolo]
mask = 3,4,5
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .5
truth_thresh = 1
random=1
```

The yolo layer invocation is 
```cpp
x = seq_imp->forward(x, inp_dim, num_classes, *_device);
```
where `inp_dim` is the image height. 
The `x` is the prediction tensor from the last final sense layer.

### Formula Implementation

`result.view(...)` formats the tensor into the below shape, so that the bounding box can easily accessed/modified.
```cpp
result = result.view({batch_size, grid_size*grid_size*num_anchors, bbox_attrs});
```

Bounding box $x$ and $y$ offset to the image are computed by
$$
\begin{align*}
    b_x &= \sigma(t_x) + c_x \\
    b_y &= \sigma(t_y) + c_y
\end{align*}
$$
```cpp
result.select(2, 0).sigmoid_();
result.select(2, 1).sigmoid_();
...
result.slice(2, 0, 2).add_(x_y_offset);
```

Bounding box width and height are computed by
$$
\begin{align*}
    b_w &= p_w e^{t_w} \\
    b_h &= p_h e^{t_h} \\
\end{align*}
$$
```cpp
result.slice(2, 2, 4).exp_().mul_(anchors_tensor);
```

The bound box has the $5$-th prediction element: if this bounding box contains an object: $\mathbb{1}^{obj}\_{ij} \in \{0,1\}$.
```cpp
result.select(2, 4).sigmoid_();
...
result.slice(2, 0, 4).mul_(stride);
```

independent logistic classifiers $\sigma\big(p_i(c)\big)$ for classification of all classes
```cpp
result.slice(2, 5, 5 + num_classes).sigmoid_();
```

### Full Code
```cpp
struct DetectionLayer : torch::nn::Module
{
	vector<float> _anchors;

    DetectionLayer(vector<float> anchors)
    {
        _anchors = anchors;
    }
    
    torch::Tensor forward(torch::Tensor prediction, int inp_dim, int num_classes, torch::Device device)
    {
    	return predict_transform(prediction, inp_dim, _anchors, num_classes, device);
    }

    torch::Tensor predict_transform(torch::Tensor prediction, int inp_dim, vector<float> anchors, int num_classes, torch::Device device)
    {
    	int batch_size = prediction.size(0);
    	int stride = floor(inp_dim / prediction.size(2));
    	int grid_size = floor(inp_dim / stride);
    	int bbox_attrs = 5 + num_classes;
    	int num_anchors = anchors.size()/2;

    	for (size_t i = 0; i < anchors.size(); i++)
    	{
    		anchors[i] = anchors[i]/stride;
    	}
    	torch::Tensor result = prediction.view({batch_size, bbox_attrs * num_anchors, grid_size * grid_size});
    	result = result.transpose(1,2).contiguous();

        // 3-d tensor by the size (batch_size) * (grid_size*grid_size*num_anchors) * (bbox_attrs)
    	result = result.view({batch_size, grid_size*grid_size*num_anchors, bbox_attrs});
    	
        // select the 2nd dim bbox_attrs
        // result.select(2, 0) and result.select(2, 1) refer to t_x and t_y
        // result.select(2, 4) refers to the logistic classifier
    	result.select(2, 0).sigmoid_();
        result.select(2, 1).sigmoid_();
        result.select(2, 4).sigmoid_();

        auto grid_len = torch::arange(grid_size);

        std::vector<torch::Tensor> args = torch::meshgrid({grid_len, grid_len});

        // flat the x and y dimensions
        torch::Tensor x_offset = args[1].contiguous().view({-1, 1});
        torch::Tensor y_offset = args[0].contiguous().view({-1, 1});

        x_offset = x_offset.to(device);
        y_offset = y_offset.to(device);

        // result.slice(2, 0, 2) selects the 2nd dim, index [0 - 2] means b_x and b_y translation of bounding boxes
        auto x_y_offset = torch::cat({x_offset, y_offset}, 1).repeat({1, num_anchors}).view({-1, 2}).unsqueeze(0);
        result.slice(2, 0, 2).add_(x_y_offset);

        torch::Tensor anchors_tensor = torch::from_blob(anchors.data(), {num_anchors, 2});
        //if (device != nullptr)
        	anchors_tensor = anchors_tensor.to(device);
        anchors_tensor = anchors_tensor.repeat({grid_size*grid_size, 1}).unsqueeze(0);

        // index [2 - 4] means b_w and b_h
        // index [5 - 5 + num_classes] means sigmoid-transformed probability to all classes
        // index [4] means the probability of the bound box contains an object
        result.slice(2, 2, 4).exp_().mul_(anchors_tensor);
        result.slice(2, 5, 5 + num_classes).sigmoid_();
   		result.slice(2, 0, 4).mul_(stride);

    	return result;
    }
};
```