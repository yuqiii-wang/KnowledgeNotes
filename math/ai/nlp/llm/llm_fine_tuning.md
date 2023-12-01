# Fine Tuning

Fine tuning is used for large model to adapt small sample data based on pre-trained parameters.

Parameter trainings:

* From scratch: totally from random parameters
* Full-parameter fine-tuning: all parameter fine-tuning
* Parameter-efficient fine-tuning: only less than $10\%$ of parameters are put in training

## LoRA: Low-Rank Adaptation of Large Language Models

For input $\bold{x} \in \mathbb{R}^{n \times d}$, where $d$ is for dimensionality, to fine tune an pretrained model, LoRA proposes below idea.
* $W_0 \in \mathbb{R}^{d \times d}$ is the pretrained parameters. 
* $A \sim N(0, \sigma^2) \in \mathbb{R}^{r \times d}$ is a weight matrix to be learned; it parameters are init by Gaussian distribution. $A$'s output reduces dimension to $r$
* $B = \bold{0} \in \mathbb{R}^{r \times d}$ is another weight matrix init to zeros. $B$'s output reset the dimension to $d$.

The training goes as below, that $W_0$ is kept unchanged/freezed; $B^{\top} A^{\top}$ are trainable parameter matrices. 

$r \ll d$ is referred to as *low-rank*.
$r=8$ is a typical implementation.
A small $r$ can help reduce computation maintaining small sizes for $A$ and $B$.

<div style="display: flex; justify-content: center;">
      <img src="imgs/lora.png" width="20%" height="20%" alt="lora" />
</div>
</br>


The new hidden layer matrix is computed as below.
$$
\bold{h} = W^{\top}_0\bold{x} + B^{\top} A^{\top} \bold{x}
$$

For intrinsic dimension (intrinsic dimension for a data set can be thought of as the number of variables needed in a minimal representation of the data), the number of neurons is small $r \ll d$ but can produce good approximation results.

##  Adapter: Parameter-Efficient Transfer Learning for NLP

Adapter adds new modules (called *adapter*) between layers of pre-trained transformer network.

Compared to LoRA, the adapter's Feed-forward down-project matrix is comparable to $A$, the up-project is comparable to $B$.
$m$ serves the same purposes as $r$'s reducing dimensions.

Adapter adds non-linearity between $A$ and $B$.

<div style="display: flex; justify-content: center;">
      <img src="imgs/petl_adapter.png" width="40%" height="50%" alt="petl_adapter" />
</div>
</br>

#### Improvement: Adapter Fusion

Adapter fusion adds attentions (value, key and query) that take adapter's output as value and key, and query from adapter's input.

Define the below parameter groups:
* Pretrained parameters $W_0$
* Adapter parameters $\Psi$
* Adapter Fusion parameters $\Phi$

The adapter fusion training goes as below:
1. fixed $W_0$, just train $\Psi$: there are multiple modules of adapters learning different knowledge
2. fixed $W_0$ and $\Psi$, train $\Phi$: attention serves as a filter that only task-specific knowledge is stored.

<div style="display: flex; justify-content: center;">
      <img src="imgs/adapter_fusion.png" width="20%" height="35%" alt="adapter_fusion" />
</div>
</br>

##  Prefix-tuning

### Prompt Tuning

Hard prompting refers to explicitly describing a task with hard-coded prompt format.

```txt
"Translate this English sentence '{How do you do ?}' into a German '{german_translation_results}'"
```

Soft prompt is trained embedding-ed input such as (in pseudo code) 

```python
soft_prompt = torch.nn.Parameter(
   torch.rand(num_tokens, embed_dim))

def input_with_soft_prompt(x, soft_prompt) :
    x = concatenate([soft_prompt, x], dim=seq_len)
    return x

train(model(input_with_soft_prompt(x, soft_prompt)))
```

### From Prompt Tuning to Prefix Tuning

The idea in prefix tuning is to add trainable tensors to each transformer block instead of only the input embeddings, as in soft prompt tuning.



<div style="display: flex; justify-content: center;">
      <img src="imgs/prefix_tuning.png" width="40%" height="30%" alt="prefix_tuning" />
</div>
</br>