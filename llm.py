from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class LocalLLM:
    def __init__(self):
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )

        tokenizer.pad_token = tokenizer.eos_token

        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.3,
            top_p=0.9
        )

    def generate(self, prompt):
        output = self.generator(prompt)[0]["generated_text"]
        return output[len(prompt):].strip()
