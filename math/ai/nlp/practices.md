# Large Language Model (LLM) Practices

model name: `gpt-3.5-turbo-16k-0613`

temperature: randomness

system: context, such as your role is a physicist,, response might be academic; the anime expert, response might be kawaii.

safety/vetting: 

## Preparation

Most popular tools to be installed:

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple datasets transformers dill tqdm ftfy
```

## Hugging Face

Hugging Face, Inc. is a French-American company and open-source community that develops tools and resources to build, deploy, and train machine learning models.

Its most famous product is `pip install transformers`.

Hugging Face provides unified APIs of different AI tasks and model configs.
These APIs can be considered wrappers of PyTorch.

### Download

Go to HuggingFace website and find desired model.
then download the files.

<div style="display: flex; justify-content: center;">
      <img src="imgs/huggingface_download.png" width="55%" height="50%" alt="huggingface_download" />
</div>
</br>

`huggingface-cli` provides bash/cmd for download by `huggingface-cli download <model_name>`.

However, in China, Hugging Face is blocked, should use a mirror site prior to downloading.

```bash
export HF_ENDPOINT=https://hf-mirror.com
```


* BERT

To download `bert-base-uncased`, there is
```bash
huggingface-cli download --resume-download bert-base-uncased --local-dir bert-base-uncased
```

where `--resume-download <model_name>` uses already downloaded `.cache` files; `--local-dir <folder_name>` puts files in the named folder.

* LLAMA

Llama needs Facebook/Meta's permission (goto Hugging Face website, login, and request for download permission (need 1 or 2 days for operation procedure process)), otherwise this error flags when downloading.

Once access granted, goto Hugging Face website, generate a token `https://huggingface.co/settings/tokens`, then use `--token hf_***` option for downloading.

```bash
huggingface-cli download --resume-download meta-llama/Llama-2-7b --token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 
```

There are many fine-tuned LLAMA variants. For example, below is open-source no need of additional download permission.

```bash
huggingface-cli download --resume-download openai-gpt --local-dir openai-gpt
```

### Pipeline

Pipeline is a convenient API that takes one argument `pipeline(task="<your-task-name>")` from the list https://huggingface.co/docs/transformers/task_summary then the underlying code helps build the relevant modules.

For example, below gives two examples, one for Q&A and one for summary.

```py
from transformers import pipeline

question_answerer = pipeline(task="question-answering")
qa_pred = question_answerer(
    question="What is the name of the repository?",
    context="The name of the repository is huggingface/transformers",
)
print(qa_pred.answer) # print "huggingface"

summarizer = pipeline(task="summarization")
sum_pred = summarizer(
    "In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles."
)
print(sum_pred.summary)
```

### `forward(...)` vs `generate(...)`

A model's `forward(...)` methods overrides the `__call()__` that serves as a go-through of a neural network.
It can be used in training as well as inference.

A model's `generate(...)` methods uses `forward(...)` as its underlying implementation.
It is only used in inference.

## Prompting Strategies

* Zero-shot prompts

Prompting the model without any example of expected behavior from the model.

```txt
Q: What is the capital of France?
A: France
```

* Few-shot prompts

Provide many examples of similar question/answer pairs as prompts, then ask the question.

```txt
Q: What is the capital of China?
A: Beijing
Q: What is the capital of Japan?
A: Tokyo
Q: What is the capital of France?
A: Paris
```

* Chain-of-thought prompts

Provide mini steps as answer rather than one simple final result such as in zero-shot.

```txt
Q: The population of China is 1.4 billions; there are 17% of population are under 14 year-old, and 9% are over 65. How many people are between 15 - 64 year-old in China ?
A: Step 1 -> 0.238 = 1.4 * 0.17; step 2 -> 0.126 = 1.4 * 0.09; step 3 -> 1.036 = 1.4 - (0.238 + 0.126); step 4 -> the population between 15 - 64 year-old is 1.036 b.
```

### LLM Tasks

* Causal Language Model (CLM)

CLM is an auto-regressive method where the model is trained in left-to-right unidirectional context to predict the next token in a sequence given the previous tokens.

OpenAI GPTs are good for such text generation and summarization.

* Seq2Seq Language Model (Seq2Seq LM)

Seq2Seq models consist of an encoder-decoder architecture, where the encoder processes the input sequence and the decoder generates the output sequence.

* Masked Language Modeling (MLM)

Some tokens in the input sequence are masked that are to be learned/predicted.

BERTs are good at this task.

* Text (Sequence/Token) Classification

From LLM outputs there are *last hidden layer outputs* and *pooler layer output*.
Last hidden layer outputs by $\text{arg max}$ can produce tokens; 
pooler layer outputs are last hidden layer outputs to fully connected layer's results, that can be used for classification tasks.

The underlying base model can be CLM or MLM.

* Question Answering

A variant of Seq2Seq LM that context texts are input to encoder and question texts are input to decoder.
The outputs/labels for training are the token positions in the context texts.

* Feature Extraction

## LangChain (LC) vs Semantic Kernel (SK)


## LLM Training Strategies/Tasks in NLP

### Optimizer Considerations

The source input of NLP is tokens from text vocabulary, and some vocabs are frequently used and some are rarely used.

By input embedding layer such as BERT base's wordpiece embedding $\bold{x}: \mathbb{R}^{1 \times 768} \rightarrow \mathbb{R}^{30522 \times 768}$ then by normalization to ${\bold{x}}_{emb-norm} \in \mathbb{R}^{1 \times 768}$ then being fed to transformer, it is 


### Auto-Regressive (AR) vs Auto-Encoding (AE)

* Auto-Regressive (AR), often used in decoder

AR simply means prediction $x_t$ by previous step input $\{x_1, x_2, ..., x_{t-1}\}$:

$$
\max_{\bold{\theta}} \log P_{\bold{\theta}} (x_1, x_2, ..., x_T) \approx
\sum_{t=1}^T \log P_{\bold{\theta}} (x_t | x_1, x_2, ..., x_{t-1})
$$

where $\bold{\theta}$ is model parameters

* Auto-Encoding (AE), often used in encoder

AE attempts to predict randomly masked tokens $\bold{x}_{\bold{m}}$ by minimizing the below objective.
The input $\bold{x}_{\overline{\bold{m}}}$ (the remaining non-masked tokens) is whole texts in which tokens are partially and randomly masked.

$$
\max_{\bold{\theta}} \log P_{\bold{\theta}} (\bold{x}_{\bold{m}} | \bold{x}_{\overline{\bold{m}}}) \approx
\sum_{t=1}^T m_t \log P_{\bold{\theta}} (x_t | \bold{x}_{\overline{\bold{m}}} )
$$

where $m_t = \left\{ \begin{array}{c} 1 & \text{the } t \text{-th token is masked} \\ 0 & \text{the } t \text{-th token is Not masked}  \end{array} \right.$ so that the log likelihood loss is only concerned with the to-be-predicted masked tokens $\bold{x}_{\bold{m}}$ .



### Training By NLP Tasks

Models are trained in different tasks to build resilience against various inputs.

* Masked language modeling (MLM)

Randomly remove words (usually $15\%$) from a sentence.
Train this model to predict the missing words.

* Next sentence prediction (NSP)

Train this model to predict a next sentence given a number of context sentences.