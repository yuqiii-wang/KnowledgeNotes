# Training Considerations

## Types of LLMs

### Chat vs Instruct

*Chat* LLM is aimed at conversations, questions and answers, back and forth - while *Instruct* LLM is for following an instruction to complete a task.

## Training Data Preparation

### Tabular Data Labelling

## Memory Consumption in Training

Reference: https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one

* $T$ = vocab/token size
* $E$ = embeddings = vocab embeddings, typically $T \times E$ forms an embedding layer, and $E = N \times D$
* $R$ = n_tr_blocks = number of transformer blocks in the model
* $N$ = n_head = number of attention heads
* $D$ = dim = dimension of each attention head
* $B$ = batch_size = batch size
* $S$ = sequence_length = input sequence length

In general,

```txt
total_memory = memory_model + memory_optimizer + memory_activations + memory_gradients + memory_overheads
```

### Model Weights:

Model parameter composition (for typical transformer structures) is 

$$
\begin{align*}
&&& \text{Embeddings} \in \mathbb{R}^{T \times E} \\
\rightarrow &&& \text{MultiHeadAttention} \in \mathbb{R}^{3 \times ( N \times D)^2} \rightarrow \text{LayerNormalization} \\
\rightarrow &&& \text{AttentionDense} \in \mathbb{R}^{1 \times ( N \times D)^2} \rightarrow \text{LayerNormalization} \\
\rightarrow &&& \text{FeedForwardDense} \in \mathbb{R}^{2 \times 4 \times ( N \times D)^2} \rightarrow \text{LayerNormalization} \\= &&& T \times E + R \times (3 + 1 + 2 \times 4) \times ( N \times D)^2
\end{align*}
$$

For example, for BERT-base, a rough estimate of model size is $108375552 = 30522 \times 768 + 12 \times 12 \times (64 \times 12)^2$, and the remaining parameters are mostly input and task specific, such as token positional embeddings pertaining to sequence length and token type embeddings indicating context/query switch.

Other examples, BERT-large parameter size estimate is $333244416 = 30522 \times 1024 + 24 \times 12 \times (64 \times 16)^2$;
T5-Small parameter size estimate is $54197760 = 32127 \times 512 + \big((12 \times 4) + (12 \times 4 \times 2)\big) \times 512^2$ (12 = 6 encoders + 6 decoders).

For parameters, there are

* 4 bytes $\times$ number of parameters for fp32 training

* 6 bytes $\times$ number of parameters for mixed precision training (maintains a model in fp32 and one in fp16 in memory)

### Optimizer States

* 8 bytes $\times$ number of parameters for normal AdamW (maintains 2 states $m_{n}$ and $v_{n}$)

$$
\begin{align*}
m_{n+1} &= \beta_1 m_{n} + (1-\beta_1) \frac{\partial\space Loss}{\partial\space W_{n}}
\\
v_{n+1} &= \beta_2 v_{n} + (1-\beta_2)\big( \frac{\partial\space Loss}{\partial\space W_{n}} \big)^2
\end{align*}
$$

* 2 bytes $\times$ number of parameters for 8-bit AdamW optimizers like bitsandbytes (1 bytes/8 bits per $m_{n}$ and $v_{n}$)

* 4 bytes $\times$ number of parameters for optimizers like SGD with momentum (maintains only 1 state $\Delta W_{n}$)

$$
\Delta W_{n+1} = \alpha \Delta W_{n} + \eta \frac{\partial\space Loss}{\partial\space W_{n}}
$$

### Gradients

* 4 bytes $\times$ number of parameters for typical fp32.

### Inputs and Forward Activations

Activations need to be stored to compute gradients.

Size depends on many factors, most important are batch size, sequence length and token dimensions.

Assume input of size `[batch_size, n_head, sequence_length, dim]`,
a general formula for memory consumption of inputs and forward activations is:

$$
\underbrace{3 \times (B \times N \times S \times D)}_{\text{Inputs: } Q, K, V} + 
\underbrace{B \times N \times S \times S}_{\text{softmax} \Big( \frac{Q K^{\top}}{\sqrt{d_k}} \Big)} + 
\underbrace{4 \times B \times N \times S \times D}_{\text{Feed Forward}}
$$

* Multi-headed Attention

$$
\text{Attention}(Q,K,V) = \text{softmax} \Big( \frac{Q K^{\top}}{\sqrt{d_k}} \Big) V
$$

where `query_layer` $Q$, `key_layer` $K$ and `value_layer` $V$ are of $\mathbb{R}^{B \times N \times S \times D}$.

Given `sequence_length` of $S$, there are

`attention_scores` $\frac{Q K^{\top}}{\sqrt{d_k}}$ yields a result of $\mathbb{R}^{B \times N \times S \times S}$.

`attention_probs` $\text{softmax} \Big( \frac{Q K^{\top}}{\sqrt{d_k}} \Big)$ yields a result of $\mathbb{R}^{B \times N \times S \times S}$.

`context_layer` $\text{softmax} \Big( \frac{Q K^{\top}}{\sqrt{d_k}} \Big) V$ yields a result of $\mathbb{R}^{B \times N \times S \times D}$.

```python
class BertSelfAttention(nn.Module):
    def forward(...):
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)

        return context_layer
```

The `context_layer` will be passed to a dense layer (such as `BertSelfOutput` for BERT) that yields a result of $\mathbb{R}^{B \times N \times S \times D}$.
This layer does not have an activation function, hence, no activation tensors to store.

```python
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```

* Feed Forward Dense

A forward pass of `intermediate` yields a result of $\mathbb{R}^{B \times N \times S \times (4 \times D)}$.

```python
intermediate_output = self.intermediate(context_layer)
```

where the `intermediate` is composed of a linear network plus activations.

```python
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
```

### Other Overheads

Some examples of overheads are attention masks, labels for loss computation, normalizations, dropouts, etc.

There are many temporary tensors during training, such as `attention_scores` and `attention_probs` from `BertSelfAttention::forward(...)` but when returned, these tensors are recycled.
Such temporary tensors are implementation-specific.

## Memory Saving Tricks in Training

### Batch Size

Implications: such batch-normalized gradients can resist extreme samples that might result in large gradients leading gradient exploding issues.
However, if batch size is too large, optimizer may stuck at local minima for gradients computed from different batches are very similar, so that the model can only learn some very generic attributes of input dataset.

Batch input can facilitate computation for samples are gradients are summed rather than computed individually.
The tensors are cache-friendly stored for they can be bulk processed.

#### Batch Size Explained

For a batch of inputs $\bold{x}_B \in \mathbb{R}^{B \times d}$, where $B$ is the number of samples in one batch; $d$ is the dimension of each sample such that $\bold{x}_i=\{ {x}_i^{(1)}, {x}_i^{(2)}, ..., {x}_i^{(d)} \}$.
The mean and standard deviation are

$$
\bold{\mu}_B = \frac{1}{B} \sum^B_{i=1} \bold{x}_i
\qquad
\bold{\sigma}_B = \frac{1}{B} \sum^B_{i=1} (\bold{x}_i - \bold{\mu}_B)^2
$$

Normalization of ${x}^{(k)}_i$ against the mean and standard deviation is 
$\hat{{x}}_i^{(k)} = \frac{{x}^{(k)}_i - {\mu}^{(k)}_B}{\sqrt{\big({\sigma}^{(k)}_B\big)^2+\epsilon}}$,
where $\epsilon=10^{-8}$ is a small value preventing zero-division error.

To use the "offsets" $\hat{\bold{x}}_B$ to the mean as the inputs, the previous non-batch transform $y_i^{(k)} = w^{(k)} x_i^{(k)} + b^{(k)}$ becomes $y_i^{(k)} = w^{(k)}_{\gamma} \hat{x}_i^{(k)} + b_{\beta}^{(k)}$.
Here defines *Batch Normalization* $BN_{w_{\gamma}^{(k)}, b_{\beta}^{(k)}}: x_{1...B}^{(k)} \rightarrow y_{1...B}^{(k)}$,
or written in function expression $\bold{y}_B^{(k)}=BN_{w_{\gamma}^{(k)}, b_{\beta}^{(k)}}(\bold{x}_B^{(k)})$

The back-propagation of the $BN_{w_{\gamma}^{(k)}, b_{\beta}^{(k)}}$ needs to compute the gradients for $w_{\gamma}^{(k)}$ and $b_{\beta}^{(k)}$ given loss $\mathcal{L}$.
The gradients are accumulated for all $B$ samples with respects to the $k$-th dimension.

$$
\frac{\partial \mathcal{L}}{\partial w_{\gamma}^{(k)}} = \sum^B_{i=1} \frac{\partial \mathcal{L}}{\partial y_i^{(k)}} \hat{x}_i^{(k)}
\qquad
\frac{\partial \mathcal{L}}{\partial b_{\beta}^{(k)}} = \sum^B_{i=1} \frac{\partial \mathcal{L}}{\partial y_i^{(k)}}
$$


#### Gradient Accumulation 

For limited memory that large batch size is not applicable, use gradient accumulation that sums up mini batches' gradients.

For example, given $\frac{\partial \mathcal{L}}{\partial w_{\gamma}^{(k)}} = \sum^B_{i=1} \frac{\partial \mathcal{L}}{\partial y_i^{(k)}} \hat{x}_i^{(k)}$, 
if $B$ is too large (a batch contains too many samples) to fit as input to model, set mini batch size to $m_j$ that they should sum up to $m_1 + m_2 + ... + m_b = B$ (mini batch sizes should be identical $m_1 = m_2 = ... = m_b$), 
then each mini batch's gradients get summed up and they are equal to the gradients by a large batch size $B$.

$$
\frac{\partial \mathcal{L}}{\partial w_{\gamma}^{(k)}} = 
\sum^{B}_{m_j=m_1} \underbrace{\Big(\sum^{m_{j+1}}_{i={m_j}} \frac{\partial \mathcal{L}}{\partial y_i^{(k)}} \hat{x}_i^{(k)} \Big)}_{\text{mini batch}} \approx
\sum^B_{i=1} \frac{\partial \mathcal{L}}{\partial y_i^{(k)}} \hat{x}_i^{(k)}
$$

Gradient accumulation can slow computation by a small degree for it includes 

### Gradient Check-Pointing

Storing all activation outputs can be very expensive for memory.
One can just discard forward activation outputs but compute on live them during back-propagation.

Generally speaking, this trick can save the memory of most activation outputs, but slow computation by $20\%$.

### Data Types

* fp32 (float32)
* fp16 (float16)
* bf16 (bfloat16)
* tf32 (CUDA internal data type)

where bf16 and tf32 are only supported Nvidia Ampere or newer (after GeForce 30 series) that see optimization dedicated to deep learning.

### Optimizers

* A standard AdamW uses 8 bytes for each parameter
* Adafactor uses slightly more than 4 bytes. Adafactor only stores aggregated information (row- and column-wise sums of the rolling averages) 
* 8bit BNB quantized AdamW, where one parameter only needs one byte 

### Use of `compute_metrics`

Metrics is used to evaluate produced results from the validation set, such as

```python
metric=load_metric("glue", "mrpc")
print(metric.inputs_description)

def compute_metrics(p: EvalPrediction):
    return metric.compute(predictions=p.predictions, references=p.labels)

trainer = Seq2SeqTrainer(
    ...
    compute_metrics=compute_metrics
)
```

In Hugging Face (by the year 2023), the `compute_metrics(p: EvalPrediction)` runs in CPU by `numpy` rather than in GPU.
Data transmission from GPU to host memory is expensive, that it occupies a huge chunk of host CPU memory and slows computation.

Try setting `eval_accumulation_steps` to a reasonable value to balance the needs of transmitting data from GPU to host memory and evaluation.

If `compute_metrics` is not set, there is no data transmission between GPU and CPU/host memory.
Instead, once training is finished, can manually choose a few good checkpoints and 