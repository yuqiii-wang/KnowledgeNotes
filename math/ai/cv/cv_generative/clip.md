# Contrastive Language-Image Pre-training (CLIP)

CLIP models are used to understand the relationship between vision features vs text description.

Typically, there are three steps:

1. Contrastive Pre-training
    * Build a similarity embedding map of text and vision features learned by contrastive loss
    * Once trained, CLIP can perform classification on unseen datasets without fine-tuning
2. Create Classifier
    * The text encoder turns these prompts into embedding vectors $T_1, T_2, ...$
3. Inference
    * When a new image is presented, the image encoder creates its embedding $I_i$.
    * The model calculates which text embedding $T_k$ is most similar to the image embedding $I_i$.

<div style="display: flex; justify-content: center;">
      <img src="imgs/clip_emb_sim.png" width="70%" height="30%" alt="clip_emb_sim" />
</div>
</br>

## CLIP Integrtaion to U-Net by Cross-Attention

In cross-attention transformer structure:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Qk^{\top}}{\sqrt{d}}\right)V
$$

* $Q$ (Query) comes from the Image (U-Net)
* $K$ (Key) and $V$ (Value) come from the Text (CLIP)


Then, the attention is projected back to the original spatial form, i.e., $W_A$ is used to transform the shape of attention matrix to fit the input shape of U-Net.

$$
\text{Attention}_{proj} =\text{Attention}\space\cdot\space W_A
$$

The projection by $W_A$ makes the text-image feature attentions map to spatial locations.

The projected attentions are added back to the original up-sampled $X_{up}^{[l]}$ (this represents the feature map in the decoder at layer $[l]$ that has just been up-sampled (resized to be larger)).

$$
X_{att}^{[l]}=X_{up}^{[l]}+\text{Attention}_{proj}
$$

The result $X_{att}^{[l]}$ of adding the projected cross-attention information back into the up-sampled decoder features.

Finally, $\hat{X}_{enc}^{[l]}$ (this represents the feature map coming from the encoder (contracting path) at the corresponding layer $[l]$) and $X_{att}^{[l]}$ are concatenated and applied convolution kernel $K$ to derive $X_o^{[l]}$.

$$
\begin{align*}
X_{concat}^{[l]} &=[\hat{X}_{enc}^{[l]};X_{att}^{[l]}] \\\\
K \otimes X_{concat}^{[l]} &= X_o^{[l]}
\end{align*}
$$

### Why $Q$ for Vision not for Text

The design of cross-attention is to let $Q$ (Query) be the center of retrieval augmentation/active search over passive info provider located by $K$ and $V$.

Given that $Q$ is for vision not for text, it is the vision data that gets concat to the feature input to U-Net and the features are guided to generate the query desired content.
