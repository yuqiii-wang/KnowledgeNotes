from transformers import pipeline
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModelForCausalLM, get_peft_config, PeftModelForSeq2SeqLM

class LLMModels:

    def __init__(self):

        model_t5_small = AutoModelForSeq2SeqLM.from_pretrained("llm_models/t5-small-lora-sum/t5-small")
        model_t5_small_lora_finetuned = PeftModelForSeq2SeqLM.from_pretrained(
            model_t5_small,
            "llm_models/t5-small-lora-sum/t5-small-lora-finetuned"
        )
        tokenizer_t5_small_lora_finetuned = AutoTokenizer.from_pretrained("llm_models/t5-small-lora-sum/t5-small-lora-finetuned")
        self.pipe_t5_small_lora_finetuned = pipeline("text2text-generation", model_t5_small_lora_finetuned, tokenizer=tokenizer_t5_small_lora_finetuned)

        model_t5_small = AutoModelForSeq2SeqLM.from_pretrained("llm_models/t5-small-lora-sum/t5-small")
        model_t5_small_prefix_finetuned = PeftModelForSeq2SeqLM.from_pretrained(
            model_t5_small,
            "llm_models/t5-small-prefix-sum"
        )
        tokenizer_t5_small_prefix_finetuned = AutoTokenizer.from_pretrained("llm_models/t5-small-prefix-sum")
        self.pipe_t5_small_prefix_finetuned = pipeline("text2text-generation", model_t5_small_prefix_finetuned, tokenizer=tokenizer_t5_small_prefix_finetuned)

        self.pipe_flan = pipeline("text2text-generation", model="llm_models/google/flan-t5-small")
        self.pipe_t5_small = pipeline("text2text-generation", model="llm_models/t5-small")

        self.pipe_model = dict()
        self.pipe_model["flan"] = self.pipe_flan
        self.pipe_model["t5_small"] = self.pipe_t5_small
        self.pipe_model["t5_small_lora_finetuned"] = self.pipe_t5_small_lora_finetuned
        self.pipe_model["t5_small_prefix_finetuned"] = self.pipe_t5_small_prefix_finetuned

    def getModel(self, name: str):
        return self.pipe_model[name]
    
    def getAllModelNames(self):
        return self.pipe_model.keys()