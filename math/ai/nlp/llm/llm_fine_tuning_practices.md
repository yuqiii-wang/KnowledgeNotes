# Parameter-Efficient Fine Tuning (PEFT) Practices


## LORA

* Task Type Options `task_type`

`CAUSAL_LM`, `FEATURE_EXTRACTION`, `QUESTION_ANS`, `SEQ_2_SEQ_LM`, `SEQ_CLS` and `TOKEN_CLS`.

## Prefix Tuning

### Prefix-Tuning on T5-Small

* set `num_virtual_tokens": num_virt_tokens` for the num of virtual tokens
* set `"token_dim": 512` for `(embed_tokens): Embedding(32128, 512)`
* set `"num_attention_heads": 8` for `(relative_attention_bias): Embedding(32, 8)` indicating there are $8$ multi-heads
* set `"encoder_hidden_size": 512` for `Linear(in_features=512, out_features=512, bias=False)`, accordingly, the dimension num of each multi-heads is $64=512/8$
* set `"num_transformer_submodules": 2` and `"num_layers": 6` for 6 encoders + 6 decoders; if `"num_transformer_submodules": 1`, there is no cross-attention and error raised "`past_key_value`" size is wrong
* set `"inference_mode": False` for training purposes
* set `"prefix_projection": False` to disable transforming the $\text{PREFIX}$ weight matrix with an additional non-linear mapping (less parameter to train, but lower accuracy)

```python
num_virt_tokens = 18

config_prefix = {
    "peft_type": "PREFIX_TUNING",
    "task_type": "SEQ_2_SEQ_LM",
    "inference_mode": False,
    "num_virtual_tokens": num_virt_tokens,
    "prefix_projection": False,
    "token_dim": 512,
    "num_transformer_submodules": 2,
    "num_attention_heads": 8,
    "num_layers": 6,
    "encoder_hidden_size": 512,
}

peft_config = get_peft_config(config_prefix)
```

### Prefix-Tuning on FLAN-T5-Small

