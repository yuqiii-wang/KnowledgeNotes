from transformers import AutoModelForSequenceClassification
from peft import PeftModelForSequenceClassification, get_peft_config



config = {
    "peft_type": "P_TUNING",
    "task_type": "SEQ_CLS",
    "inference_mode": False,
    "encoder_reparameterization_type": "MLP",
    "num_virtual_tokens": 20,
    "token_dim": 768,
    "num_transformer_submodules": 1,
    "num_attention_heads": 12,
    "num_layers": 12,
    "encoder_hidden_size": 768,
}

peft_config = get_peft_config(config)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
peft_model_prefix = PeftModelForSequenceClassification(model, peft_config)
peft_model_prefix.print_trainable_parameters()
print(peft_model_prefix)


#########

config = {
    "peft_type": "PREFIX_TUNING",
    "task_type": "SEQ_CLS",
    "inference_mode": False,
    "num_virtual_tokens": 20,
    "token_dim": 768,
    "num_transformer_submodules": 1,
    "num_attention_heads": 12,
    "num_layers": 12,
    "encoder_hidden_size": 768,
    "prefix_projection": False,
}

peft_config = get_peft_config(config)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
peft_model_prefix = PeftModelForSequenceClassification(model, peft_config)
peft_model_prefix.print_trainable_parameters()
print(peft_model_prefix)


#########

config = {
    "peft_type": "PROMPT_TUNING",
    "task_type": "SEQ_CLS",
    "inference_mode": False,
    "num_virtual_tokens": 20,
    "prompt_tuning_init": "PromptTuningInit.TEXT",
    "prompt_tuning_init_text": "Classify if this text's sentiment is a complaint or not:",
    "token_dim": 768,
    "num_transformer_submodules": 1,
    "num_attention_heads": 12,
    "num_layers": 12,
}

peft_config = get_peft_config(config)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
peft_model_prompt = PeftModelForSequenceClassification(model, peft_config)
peft_model_prompt.print_trainable_parameters()
print(peft_model_prompt)


#########

config = {
    "peft_type": "LORA",
    "task_type": "SEQ_CLS",
    "inference_mode": False,
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "fan_in_fan_out": False,
    "bias": "none",
}
peft_config = get_peft_config(config)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
peft_model_lora = PeftModelForSequenceClassification(model, peft_config)
peft_model_lora.print_trainable_parameters()
print(peft_model_lora)
