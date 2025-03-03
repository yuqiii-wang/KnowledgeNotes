# Embeddings

Embedding describes information representation and compression, representing a token/word as a vector.

## Semantics/Linguistics

For example, the word "restaurants" has the below attributes:

* isNoun: $\{0, 0, 0, 1, 0\}$ for $\{\text{isVerb}, \text{isAdjective}, \text{isPronoun}, \text{isNoun}, \text{isAdverb}\}$
* isPlural: $\{1\}$ for $\{\text{isPlural}\}$  
* synonyms: $\{ 5623, 1850, 2639 \}$ (vocabulary index) for $\{ \text{hotel}, \text{bar}, \text{club} \}$
* antonyms: $\emptyset$
* frequent occurrences under what topics: $\{ 1203, 5358, 1276 \}$ (vocabulary index) for $\{ \text{eating}, \text{outing}, \text{gathering} \}$
* Term frequency-inverse document frequency (TF-IDF): $\{ 0.016, 0.01, 0.0, 0.005 \}$ , formula:
  * $\text{TF-IDF}_j = \text{Term Frequency}_{i,j} \times \text{Inverse Document Frequency}_{i}$, where
  * $\text{Term Frequency}_{i,j} = \frac{\text{Term i frequency in document j}}{\text{Total no. of terms in document j}}$
  * $\text{Inverse Document Frequency}_{i} = \log \frac{\text{Total no. of documents}}{\text{No. of documents containing term i}}$

Given the four sentences/documents,

```txt
There are many popular restaurants nearby this church.
Some restaurants offer breakfasts as early as 6:00 am to provide for prayers.
"The price and taste are all good.", said one prayer who has been a frequent visitor to this church since 1998.
However, Covid-19 has forced some restaurants to shut down for lack of revenue during the pandemic, and many prayers are complained about it.
```

The TF-IDF per sentence/document is computed as below.

|No.|Token|Term count (Doc 1)|Term count (Doc 2)|Term count (Doc 3)|Term count (Doc 4)|Document count|IDF|TF $\times$ IDF (Doc 1)|TF $\times$ IDF (Doc 2)|TF $\times$ IDF (Doc 3)|TF $\times$ IDF (Doc 4)|
|-|-|-|-|-|-|-|-|-|-|-|-|
|1|many|0.125|0|0|0.043478260869565216|2|0.301|0.038|0|0|0.013|
|2|popular|0.125|0|0|0|1|0.602|0.075|0|0|0|
|3|restaurants|0.125|0.07692307692307693|0|0.043478260869565216|3|0.125|0.016|0.01|0|0.005|
|4|nearby|0.125|0|0|0|1|0.602|0.075|0|0|0|
|5|church|0.125|0|0.047619047619047616|0|2|0.301|0.038|0|0.014|0|
|6|offer|0|0.07692307692307693|0|0|1|0.602|0|0.046|0|0|

For compression, one popular approach is encoder/decoder, where dataset is fed to machine learning study.

For example, by placing "restaurants" and "bar" together in a text dataset that only describes food, likely the attribute "topic" might have little information hence neglected (set to zeros) in embeddings.

## Rotational Position Embeddings (RoPE)

https://zhuanlan.zhihu.com/p/662790439

Positional embeddings represent the position of a word in a sentence/document.
The order of how vocabularies are arranged in a sentence/document provides rich information in NLP.

Transformer uses the below formulas to compute positional embeddings (PE) by a rotation matrix $R(\bold{\theta}_i)$ where each token position is represented/offset by $\bold{\theta}_i$ with respect to dimension $\bold{d}$.

$$
\begin{align*}
\text{PE}(i) &= R (\bold{\theta}_i)
\qquad
\text{where } \bold{\theta}_i = 10000^{-\frac{2i}{\bold{d}}}
\end{align*}
$$

where $\bold{d}=\{ 1,2,...,D \} \in \mathbb{Z}^{+}$ is a vector of dimension indices, then define $\bold{\theta}_i=10000^{-\frac{2i}{\bold{d}}}$, where $\bold{\theta}_i = \{ {\theta}_{i_{1}}, {\theta}_{i_{2}}, ..., {\theta}_{i_{D}} \}$,
and $i \in \mathbb{Z}^{+}$ is the position of a word in a sentence/document.

## RoPE Derivation

### Linear Position Embedding

Define a score to be maximized when query $\bold{q}_i$ is positionally "close" to key $\bold{k}_j$.
The $i$ and $j$ individually represent the positions of query and key in a sentence/document, hence $i-j$ represents the relative position gap.

$$
\max \text{score}(\bold{q}_i, \bold{k}_j) =
(\bold{q}_i + \bold{p}_{i-j})^{\top} (\bold{k}_j + \bold{p}_{i-j}) - \bold{p}^{\top}_{i-j} \bold{p}_{i-j}
$$

where $\bold{p}_{i-j}$ serves as a linear relative position gap.

This design's motivation is that in NLP, if a query word is adjacent to a key word, they should be highly semantically related.
Their multiplication value should be large (this $\text{score}(\bold{q}_i, \bold{k}_j)$ is named *attention score* in transformer), otherwise small, so that attention mechanism can easily produce differences during matrix multiplication in this regard.

### Position Embedding by Rotation Matrix

Here uses sinusoid to represent the relative position gap by a rotation matrix $R_{i-j}$ to replace the above linear position gap $\bold{p}_{i-j}$.
Sinusoid not only decreases fast in $\text{score}(\bold{q}_i, \bold{k}_j)$ as positional gap grows against linear decrease by $\bold{p}_{i-j}$, but also has sinusoidal patterns that recursively see highs and lows in different relative position gaps $|i-j|$ with respects to different dimensions $d$.

Set $\bold{q}_i=R_{i}\bold{q}_1$ and $\bold{k}_j=R_{j}\bold{k}_1$ so that their position info is represented via rotation matrices $R_{i}$ and $R_{j}$, there is

$$
\max \text{score}(\bold{q}_i, \bold{k}_j) =
(R_{i} \bold{q}_1)^{\top} (R_{j} \bold{k}_1) =
\bold{q}_1^{\top} R_{i}^{\top}  R_{j} \bold{k}_1 =
\bold{q}_1^{\top} R_{i-j} \bold{k}_1
$$

Now use and $\theta_i \in (10^{-4}, 1]$ such that $\theta_i=10000^{-\frac{2i}{\bold{d}}}$ to assign discrete values to $R_{i-j}$.

Let $D$ represent the dimension num of $\bold{v}_i \in \mathbb{R}^{1 \times D}$.
Let $R(\theta)$ be a rotation matrix for a vector $\bold{v}_i$, there is

$$
\cos(\theta) = \frac{\bold{v}_i \cdot \bold{v}_j}{||\bold{v}_i || \space || \bold{v}_j ||}
\qquad
R (\theta) = \begin{bmatrix}
      \cos \theta & -\sin \theta \\
      \sin \theta & \cos \theta \\
\end{bmatrix}
$$

Rotation relative info can be computed by $R_{\theta_{i}-\theta_{j}}=R_{\theta_{i}}^{\top}{R_{\theta_{j}}}$, there is

$$
R(\theta) = \begin{bmatrix}
    \cos \theta_1 & -\sin \theta_1 & 0 & 0 & & & 0 & 0 \\
    \sin \theta_1 & \cos \theta_1 & 0 & 0 & & & 0 & 0 \\
    0 & 0 & \cos \theta_2 & -\sin \theta_2 & & & 0 & 0 \\
    0 & 0 & \sin \theta_2 & \cos \theta_2 & & & 0 & 0 \\
    & & & & \ddots & \ddots & & & \\
    & & & & \ddots & \ddots & & & \\
    0 & 0 & 0 & 0 & & & \cos \theta_{D/2} & -\sin \theta_{D/2} \\
    0 & 0 & 0 & 0 & & & \sin \theta_{D/2} & \cos \theta_{D/2} \\
\end{bmatrix}
$$

If $\bold{v} \in \mathbb{R}^{2 \times D}$, there is

$$
\begin{align*}
  R(\theta) \bold{v} &=
  \begin{bmatrix}
      \cos \theta & -\sin \theta \\
      \sin \theta & \cos \theta \\
  \end{bmatrix}
  \begin{bmatrix}
      \bold{v}_1 \\
      \bold{v}_2 \\
  \end{bmatrix}
\\ &=
  \begin{bmatrix}
      \bold{v}_1 \cos \theta - \bold{v}_2 \sin \theta \\
     \bold{v}_1  \sin \theta + \bold{v}_2 \cos \theta \\
  \end{bmatrix}
\\ &=
\begin{bmatrix}
      \bold{v}_1 \\ \bold{v}_2 
\end{bmatrix} \odot
\begin{bmatrix}
      \cos \theta \\ \cos \theta
\end{bmatrix} +
\begin{bmatrix}
      \bold{v}_1 \\ \bold{v}_2
\end{bmatrix} \odot
\begin{bmatrix}
      -\sin \theta \\ \sin \theta
\end{bmatrix}
\end{align*}
$$

where $\odot$ is element-wise multiplication operator.

If $\bold{v} \in \mathbb{R}^{n \times D}$, where $n$ is the num of tokens
Here sets $n=D$, there is

$$
R(\theta) \bold{v} =
\begin{bmatrix}
      \bold{v}_1 \\ \bold{v}_2 \\ \bold{v}_3 \\ \bold{v}_4 \\ \vdots \\ \bold{v}_{D-1} \\ \bold{v}_{D}
\end{bmatrix} \odot
\begin{bmatrix}
      \cos \theta_1 \\ \cos \theta_1  \\ \cos \theta_2 \\ \cos \theta_2 \\ \vdots \\ \cos \theta_{D/2} \\ \cos \theta_{D/2}
\end{bmatrix} +
\begin{bmatrix}
      \bold{v}_1 \\ \bold{v}_2 \\ \bold{v}_3 \\ \bold{v}_4 \\ \vdots \\ \bold{v}_{D-1} \\ \bold{v}_{D}
\end{bmatrix} \odot
\begin{bmatrix}
      -\sin \theta_1 \\ \sin \theta_1  \\ -\sin \theta_2 \\ \sin \theta_2 \\ \vdots \\ -\sin \theta_{D/2} \\ \sin \theta_{D/2}
\end{bmatrix}
$$

For a query token $\bold{q}_{1}$ (set $i=1$ as base position reference index since only relative positional gap is concerned here), plot score $\text{score}(\bold{q}_1, \bold{k}_j)$ for key tokens at growing distances $\bold{k}_{j}$ for $j=1,2,...,512$ and $j=1,2,...,65535$.
These two plots show the scores as a key $\bold{k}_j$'s positional distance ($\text{dist}=|i-j|$) to $\bold{q}_{1}$ grows.

For easy comparison, both query and key token are set to $\bold{1}$ such that $\bold{q}_i=\{ \underbrace{1,1,1, ..., 1 }_{D=256} \}$ and $\bold{k}_j=\{ \underbrace{1,1,1, ..., 1 }_{D=256} \}$, so that the scores' differences only reflect the rotational positional distance ($\bold{q}_1^{\top} R_{1-j} \bold{k}_j$).

By matrix multiplication, score is the sum of each dimension's multiplication result divided by a normalization term $\sqrt{D}$ such that $\text{score}(\bold{q}_i, \bold{k}_j)=\bold{q}_1^{\top} R_{i-j} \bold{k}_1= \frac{1}{\sqrt{D}} \sum^D_{d=1} q_{1_{d}}^{\top} R_{(1-j)_{d}} {k}_{1_{d}}$.

The individual dimensions' scores for $i=1$ vs $j$ are shown as below.

* for low dimensions, sinusoids see complete cycles for small $|i-j|$;
* for high dimensions, sinusoids see complete cycles for large $|i-j|$.

By this design, each position embedding dimension learns about info regarding different $|i-j|$.

<div style="display: flex; justify-content: center;">
      <img src="imgs/rope_query0_keyj_individuald.png" width="80%" height="20%" alt="rope_query0_keyj_individuald" />
</div>
</br>

For summed score over all dimensions $\text{score}(\bold{q}_i, \bold{k}_j)=\frac{1}{\sqrt{D}} \sum^D_{d=1} q_{1_{d}}^{\top} R_{(1-j)_{d}} {k}_{1_{d}}$, 

* when they are close (small values of $|i-j|$), score is high;
* when they are far away (large values of $|i-j|$), score is low.

Figure 1 is the zoomed-in version of figure 2 for $j=1,2,...,512$.

<div style="display: flex; justify-content: center;">
      <img src="imgs/rope_query0_keyj.png" width="50%" height="25%" alt="rope_query0_keyj" />
</div>
</br>

### RoPE Example and Frequency Study

* Let $D=16$ and $D/2=8$.
* Frequency base $10,000$
* Rotation angle for for dim $d$: $\theta_d = 10000^{-\frac{2d}{D}}$
* Assume (as an example) a query vector $\bold{q}_m$ sees a key vector $\bold{k}_n$ at $n=m+3$ (relative position distance is $3$)

Compute the angles and their cosine and sine values:

$$
\begin{align*}
    \theta_0 &= 10000^{-\frac{2 \times 0}{16}} = 10000^0 = 1 &\qquad
    \cos(\theta_0) &\approx 0.5403 & \sin(\theta_0) &= 0.8418 \\
    \theta_1 &= 10000^{-\frac{2 \times 1}{16}} = 10000^{-\frac{1}{8}} \approx 0.3162 &\qquad
    \cos(\theta_1) &\approx 0.9504 & \sin(\theta_1) &\approx 0.3110 \\
    \theta_2 &= 10000^{-\frac{2 \times 2}{16}} = 10000^{-\frac{1}{4}} = 0.1 &\qquad
    \cos(\theta_2) &\approx 0.9950 & \sin(\theta_2) &\approx 0.0998 \\
    \theta_3 &= 10000^{-\frac{2 \times 3}{16}} = 10000^{-\frac{3}{8}} \approx 0.03162 &\qquad
    \cos(\theta_3) &\approx 0.9996 & \sin(\theta_3) &\approx 0.0316 \\
    \theta_4 &= 10000^{-\frac{2 \times 4}{16}} = 10000^{-\frac{1}{2}} = 0.01 &\qquad
    \cos(\theta_4) &\approx 0.99995 & \sin(\theta_4) &\approx 0.0099998 \\
    \theta_5 &= 10000^{-\frac{2 \times 5}{16}} = 10000^{-\frac{5}{8}} \approx 0.003162 &\qquad
    \cos(\theta_5) &\approx 1 & \sin(\theta_5) &\approx 0 \\
    \theta_6 &= 10000^{-\frac{2 \times 6}{16}} = 10000^{-\frac{3}{4}} = 0.001 &\qquad
    \cos(\theta_6) &\approx 1 & \sin(\theta_6) &\approx 0 \\
    \theta_7 &= 10000^{-\frac{2 \times 7}{16}} = 10000^{-\frac{7}{8}} \approx 0.000316 &\qquad
    \cos(\theta_7) &\approx 1 & \sin(\theta_7) &\approx 0 \\
\end{align*}
$$

For $\bold{v} \in \mathbb{R}^{16}$, group by pairing

$$
\text{Groups}=(v_1, v_2), (v_3, v_4), ..., (v_{15}, v_{16})
$$

Compute the distance at the position $m$ for each group by rotation

$$
\begin{bmatrix}
    v_{2i} \cos(m \theta_i) - v_{2i+1} \sin(m \theta_i) \\
    v_{2i} \sin(m \theta_i) + v_{2i+1} \cos(m \theta_i)
\end{bmatrix}
$$

Given the assumption that query vector $\bold{q}_m$ sees key vector $\bold{k}_n$
at $n=m+3$ (relative position distance is $3$), compute the score by rotation (replace $\bold{v}$ with $\bold{q}_m$ and $\bold{k}_n$).

$$
\begin{align*}
\langle \bold{q}_m, \bold{k}_n \rangle = \sum_{i=0}^7 \Big(
    & \underbrace{\big(q_{2i}^m k_{2i}^{m+3} + q_{2i+1}^m k_{2i+1}^{m+3}\big)}_{\alpha_{\cos}} \cos(3\theta_i) + \\
    & \underbrace{\big(q_{2i+1}^m k_{2i}^{m+3} - q_{2i}^m k_{2i+1}^{m+3}\big)}_{\alpha_{\sin}} \sin(3\theta_i) \Big)
\end{align*}
$$

The above formula shows that for relative position distance $3$,
it is cosine and sine of $3\theta_i$ that contribute to the score.

#### RoPE Frequency Study

Generally speaking, low frequency groups see higher variations as positional distance grows,
while high frequency groups see lower variations.

##### High Frequency Groups

For large $\theta_i$, $\cos(3\theta_i)$ and $\sin(3\theta_i)$ see large change from $\cos(1\theta_i)$ and $\sin(1\theta_i)$ respectively.

$$
\begin{align*}
\cos(1\theta_0) &\approx 0.5403 & \sin(1\theta_0)       &\approx 0.8418 &\qquad\Rightarrow\qquad
\cos(3\theta_0) &\approx -0.9900 & \sin(3\theta_0) &\approx 0.1411 \\
\cos(1\theta_1) &\approx 0.3162 & \sin(1\theta_1)  &\approx 0.9594 &\qquad\Rightarrow\qquad
\cos(3\theta_1) &\approx -0.5828 & \sin(3\theta_1) &\approx 0.8126 \\
\end{align*}
$$

##### Low Frequency Groups

For small $\theta_i$, $\cos(3\theta_i)$ and $\sin(3\theta_i)$ see small change from $\cos(1\theta_i)$ and $\sin(1\theta_i)$ respectively.

$$
\begin{align*}
\cos(1\theta_6) &\approx 1 & \sin(1\theta_6)  &\approx 0 &\qquad\Rightarrow\qquad
\cos(3\theta_6) &\approx 1 & \sin(3\theta_6) &\approx 0 \\
\cos(1\theta_7) &\approx 1 & \sin(1\theta_7)  &\approx 0 &\qquad\Rightarrow\qquad
\cos(3\theta_7) &\approx 1 & \sin(3\theta_7) &\approx 0 \\
\end{align*}
$$

#### RoPE Long Distance Study

Let $\lambda_i=\frac{2\pi}{\theta_i}=2\pi \cdot 10000^{2i/16}$ be wavelength.

|$\theta_i$|Full wavelength $\lambda_i$|Half wavelength $\lambda_i/2$|
|-|-|-|
|$\theta_0$=1|6.2832|3.1416|
|$\theta_1$=0.3162|19.9477|9.9738|
|$\theta_2$=0.1|62.832|31.416|
|$\theta_3$=0.03162|199.477|99.738|
|$\theta_4$=0.01|628.32|314.16|
|$\theta_5$=0.003162|1994.77|997.38|
|$\theta_6$=0.001|6283.2|3141.6|
|$\theta_7$=0.0003162|19947.7|9973.8|

The max half wavelength $\lambda_i/2=9973.8$ by the highest frequency $\theta_7$ means theoretical max token length (context length),
that by incremental rotation of $9973.8$ times so that a half wavelength $\pi$ is covered.

However, this is not advised because only the highest frequency dimension by $\theta_7$ can cover the whole $\pi$ area, and lower frequency dimensions are totally lost.

If exceeded the wavelength, for $\alpha_{\cos}>\alpha_{\sin}$, the attention score $\langle \bold{q}_m, \bold{k}_n \rangle$ is dominated by the cosine, that has mirror values when $\Delta\theta_i>\pi$, where $\Delta>9973.8$, i.e., there are multiple mappings of queries vs keys given an attention score by $\bold{q}^{\top}_i\bold{k}_j$.

##### Frequency Study In Long Distance

When two tokens are very distant $|n-m|=\Delta\rightarrow \infty$, the score $\langle \bold{q}_m, \bold{k}_n \rangle$ has multiple mappings hence the attention score cannot determine which query token be associated to which key token.
Consequently, in long distance, the attention mechanism fails.

Consider

$$
\begin{align*}
\langle \bold{q}_m, \bold{k}_n \rangle = \sum_{i=0}^{D/2-1} \Big(
    & \underbrace{\big(q_{2i}^m k_{2i}^{m+\Delta} + q_{2i+1}^m k_{2i+1}^{m+\Delta}\big)}_{\alpha_{\cos}} \cos(\Delta\theta_i) + \\
    & \underbrace{\big(q_{2i+1}^m k_{2i}^{m+\Delta} - q_{2i}^m k_{2i+1}^{m+\Delta}\big)}_{\alpha_{\sin}} \sin(\Delta\theta_i) \Big)
\end{align*}
$$

To study the series $\sum_{i=0}^{D/2-1}\big(\alpha_{\cos}\cos(\Delta\theta_i)+\alpha_{\sin}\sin(\Delta\theta_i)\big)$ as $\Delta\rightarrow\infty$,
first consider this expression $\cos(\Delta\theta_i)+\sin(\Delta\theta_i)$,

* $\cos(\Delta\theta_i)$ and $\sin(\Delta\theta_i)$ are oscillation function (does not converge to a limit but oscillate within a range)
* $\cos(\Delta\theta_i)$ and $\sin(\Delta\theta_i)$ linear combinations are also oscillating.

One can prove that

$$
\begin{align*}
    \max\big(\cos(\Delta\theta_i)+\sin(\Delta\theta_i)\big)&=\sqrt{2} \\
    \min\big(\cos(\Delta\theta_i)+\sin(\Delta\theta_i)\big)&=-\sqrt{2} \\
\end{align*}
$$

As a result, the convergence behavior of $\langle \bold{q}_m, \bold{k}_n \rangle$ is determined by its linear coefficients $\alpha_{\cos}$ and $\alpha_{\sin}$.
Further more, for $\alpha_{\cos}>\alpha_{\sin}$, the convergence behavior is dominated by the cosine term.

For a very large $\Delta\rightarrow\infty$, the $\Delta\theta_i$ steps across multiple oscillation ranges $[[0, \pi), [\pi, 2\pi), [2\pi, 3\pi), ...]$,
so that the attention score cannot determine which query token be associated to which key token.

In more detail, for high frequency sequence the $\cos(\Delta\theta_{\text{large}})$

## Embedding by Deep Learning

Above embedding designs contain rich semantics.
However, embeddings can be trained by large corpus such as by BERT base there is embeddings $W_{EmbBertBase} \in \mathbb{R}^{30522 \times 768}$ corresponding to $30522$ tokens.

Tokens are assigned a id such as $2023$ for "This", and it is one-hot encoded.

$$
\bold{t}_{i=2023} = \text{OneHotEncoding}(2023) =
[\underbrace{0, 0, 0, 0, 0, 0, 0, 0, ..., 0}_{\times 2022}, 1, \underbrace{0, 0, 0, 0, 0, 0, 0, 0, ..., 0}_{\times (30522 - 1 - 2022)}]
$$

By matrix multiplication, the token's embedding is retrieved from $E_{emb}$.

$$
W_{EmbBertBase}^{\top} \bold{t}_i \in \mathbb{R}^{1 \times 768}
$$

<div style="display: flex; justify-content: center;">
      <img src="imgs/tokenization_then_embedding.png" width="30%" height="40%" alt="tokenization_then_embedding" />
</div>
</br>

In Hugging face, there is

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

embedding_matrix = model.embeddings.word_embeddings.weight
embedding_this = embedding_matrix[2023]
print(embedding_this.size()) # print "torch.Size([768])"
print(embedding_matrix[2023][:10]) # top 10 embs are tensor([-0.0571,  0.0153, -0.0047,  0.0105, -0.0279,  0.0218, -0.0006,  0.0224,
                                   # 0.0225,  0.0135], grad_fn=<SliceBackward0>)
```
