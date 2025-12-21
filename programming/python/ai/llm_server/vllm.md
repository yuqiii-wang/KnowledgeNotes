# LLM Hosting by VLLM

## Quick Start: VLLM Docker Deployment with Downloaded Model

For example, first download model to `"$(pwd)"/models/Qwen3-14B-FP8`, then run

```sh
docker run -d \
  --gpus all \
  --name vllm-qwen3-14b-fp8 \
  -v "$(pwd)"/models/Qwen3-14B-FP8:/root/Qwen3-14B-FP8:ro \
  -p 8001:8000 \
  egs-registry.cn-hangzhou.cr.aliyuncs.com/egs/vllm:0.11.0-pytorch2.8-cu128-20251015 \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /root/Qwen3-14B-FP8 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --disable-custom-all-reduce \
    --gpu-memory-utilization 0.9 \
    --kv-cache-dtype fp8 \
    --port 8000 \
    --max_model_len 32768
```

where `-v "$(pwd)"/models/Qwen3-14B-FP8:/root/Qwen3-14B-FP8:ro` sets up a virtual disk mapping host machine `"$(pwd)"/models/Qwen3-14B-FP8` to docker container dir `/root/Qwen3-14B-FP8`, then `python3 -m vllm.entrypoints.openai.api_server` will read the model `--model /root/Qwen3-14B-FP8` via container-host mapped disk dir loading the downloaded `"$(pwd)"/models/Qwen3-14B-FP8`.

The other vLLM args are

* `--trust-remote-code`: Required for Qwen. Qwen models usually contain custom Python code (in modeling_qwen.py) that isn't built into the standard Transformers library yet. This flag gives permission to execute that code.
* `--tensor-parallel-size 1`: Tells vLLM to use exactly 1 GPU; if there are multiple GPUs, set it to the required num accordingly.
* `--disable-custom-all-reduce`: Disables a specific optimized kernel for cross-GPU communication. This is set up for  `--tensor-parallel-size 1`
* `--gpu-memory-utilization 0.9`: Tells vLLM: "Reserve 90% of the GPU's total VRAM."
* `--kv-cache-dtype fp8`: Stores the Context Memory (KV Cache) in FP8 (1 byte) instead of standard FP16 (2 bytes).
* `--max_model_len 32768`: Max context length

P.S., `modeling_qwen.py` is the architectural blueprint of the model written in PyTorch, used for "Qwen3" (which is likely a very new by 2025 or preview version not yet built into the standard Hugging Face transformers library), this file contains the custom code required to run it.

## Memory Consumption

### Total Memory Equation

#### Model static memory

* Attention Weights: $W_Q, W_K, W_V, W_O$
* MLP (Feed Forward) Weights: $3 \times \text{IntermediateSize} \times \text{HiddenSize}$ for Gate, Up, Down matrices (SwiGLU uses 3 matrices)
* Embeddings: $2 \times \text{VocabSize} \times \text{HiddenSize}$, where $2$ is for input and output mapping
* Precision: FP16/BF16 = 2 bytes, FP8 = 1 byte, INT4 = 0.5 bytes
* Some overheads: slightly proportional to model size

#### Token KV Cache Memory

$$
\text{OneTokenMem} = 2 \times \text{Layer} \times \text{KVHeads} \times \text{HeadsDim} \times \text{Precision}
$$

where

* $\text{Layer}$ = Number of Layers (`num_hidden_layers`)
* $\text{HeadsDim}$ = Size of one attention head (`head_dim`)
* $\text{KVHeads}$ =  = Number of Key/Value Heads (`num_key_value_heads`).
* $\text{Precision}$ = Precision size in bytes (FP16/BF16 = 2 bytes, FP8 = 1 byte, INT4 = 0.5 bytes).

The total token cache is $\text{TotalTokenMem} = \text{TokenLength} \times \text{OneTokenMem}$

#### Remaining Memory for Request Concurrency

The memory for actual user request is $\text{TotalVllmGPUMem}-\text{ModelStaticMem}$.
Then, the concurrency is $\frac{\text{TotalVllmMem}\space-\space\text{ModelStaticMem}}{\text{TotalTokenMem}}$

### Example: QWen3-14B-FP8

```json
{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 17408,
  "max_position_embeddings": 40960,
  "max_window_layers": 40,
  "model_type": "qwen3",
  "num_attention_heads": 40,
  "num_hidden_layers": 40,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936,
  "quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [
      128,
      128
    ]
  }
}
```

#### Model static memory

* Attention Weights

$$
\begin{align*}
W_Q &: 5120 \times 5120 = 26.2 \text{mil} \\
W_K &: 5120 \times (8 \times 128) = 5.2 \text{mil} \\
W_V &: 5120 \times 1024 = 5.2 \text{mil} \\
W_O &: 5120 \times 5120 = 26.2 \text{mil} \\
\end{align*}
$$

One attention block has $62.9 \text{mil} = 26.2 \text{mil} + 5.2 \text{mil} + 5.2 \text{mil} + 26.2 \text{mil}$

* MLP (Feed Forward) Weights

$$
\text{MLP}: 3 \times (5120 \times 17408) = 267.4 \text{mil}
$$

Given a total of $40$ layers, there is $13.21 \text{bil} = 40 \times (62.9 \text{mil} + 267.4 \text{mil})$

* Input and Output Embedding

For `tie_word_embeddings=False`, only need to count input and output embedding

$$
1.56 \text{bil} = 2 \times 151,936 \times 5120
$$

Total Parameter Count is $14.77 \text{bil} = 1.56 \text{bil} + 13.21 \text{bil}$.

P.S.,  `tie_word_embeddings` is a configuration setting that decides whether the model should use **the same memory matrix** for the beginning (input embedding) and the end of its processing (output embedding).

If `tie_word_embeddings=True`, the model forces the Input Matrix and the Output Matrix to be identical, and the learned vector representations (semantics) of tokens are very similar between input vs output.
However, modern research (like Qwen and Llama 3) found that untying them leads to slightly better performance, especially for multilingual models or models with massive vocabularies.

* Precision: FP8 (defined by `quant_method: "fp8"` and `fmt: "e4m3"`)

$$
14.77 \text{bil bytes} = 14.77 \text{bil} \times 1 \text{bytes}
$$

Consider about $1\text{GB}$ overheads, the total model static memory is

$$
15.77\text{BG} = 14.77 \text{bil bytes} + 1\text{GB}
$$

##### Overheads in the 1GB

P.S., overheads refer to (slightly proportional to model size) static or dynamic but marginal memory costs:

1) The cost of simply initializing the NVIDIA driver and PyTorch on the GPU.
2) vLLM pre-compiles the computational graph (like a map of the math it intends to do) to make execution faster. It reserves memory for this map.
3) vLLM page index table

It is known that vLLM manages KV cache as blocks indexed by a page table to achieve fast KV retrieval.

For example, for `max_model_len: 32,768` tokens and `block_size: 16` (Default vLLM setting. 1 Block = 16 tokens), and `max_num_seqs: 256` (Default vLLM limit for concurrent requests, a.k.a., batch size) and `dtype: Int32` (4 Bytes as pointer address), there is

$$
\text{BlockPerRequest}=\frac{\text{ContextLength}}{\text{BlockSize}}=\frac{32768}{16}=2048
$$

vLLM pre-allocates enough space for the worst-case scenario where every slot in the batch is full.

$$
\text{TotalSlots}=\text{BatchSize}\times\text{BlockPerRequest}=256\times 2048 = 524288
$$

There $524,288$ entries that need indexing, so that the pointer address size is $2\text{MB} = 524288 \times 4 \text{bytes}$

4) "Scratchpad." When the model multiplies matrices, it needs a place to store the intermediate results (e.g., the output of Layer 1 before it goes into Layer 2); the wider a layer is, the larger the scratchpad is.

|Model Size Class|Typical Hidden Size|Estimated Overhead|
|:---|:---|:---|
|Small (0.5B - 3B)|1024 - 2560|~0.8 GB|
|Medium (7B - 14B)|4096 - 5120|~1.2 GB - 1.8 GB|
|Large (30B - 72B)|6144 - 8192|~2.0 GB - 3.5 GB|
|Massive (405B+)|16384+|~5.0 GB+|

#### Token KV Cache Memory

$$
2 \times \text{Layer} \times \text{KVHeads} \times \text{HeadsDim} \times \text{Precision} = \\
\text{OneTokenMem} = 0.078125 \text{MB} = 2 \times 40 \times 8 \times 128 \times 1 \text{byte}
$$

For a `"max_position_embeddings": 40960`, the cache mem is $3.125\text{GB} = 40960 \times 0.078125 \text{MB}$.

However, the claimed $40960$ is model max context capability, in practice there sets $80\%$ of the max context, i.e., $32768 = 40960 \times 0.8$.
Accordingly, the cache mem is $2.5\text{GB} = 32768 \times 0.078125 \text{MB}$.

##### The Input and Output Length in Token Context

The context length of $32768$ tokens need to get shared between  input and output, i.e., $32768=\text{InputLength}+\text{OutputLength}$.
The $\text{InputLength}$ is also known as *prompt*.
If user considers the output is long, this user needs to make sure his/her prompt be short.

#### Remaining Memory for Request Concurrency

Before jumping in processing user request, on starting VLLM, there usually specifies `--gpu-memory-utilization 0.9` that is used to limit the amount of mem could be used for VLLM.
The `0.9` gives $10\%$ buffer for GPU to process unexpected workload, e.g., some apps or browser windows may request to use GPU for rendering.  
As a result, $\text{TotalVllmMem} = 0.9 \times \text{TotalGPUMem}$.

Consider $\text{TotalGPUMem}=32\text{GB}$, the $\text{TotalVllmMem}=28.8\text{GB}$. The max concurrency is $5.21=\frac{28.8-15.77}{2.5}$.
If user input is short, there could be much more concurrent requests; the hard limit is all summed user sessions should NOT exceed the free mem $13\text{GB}=28.8-15.77$.
