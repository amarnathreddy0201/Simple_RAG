# from transformers import pipeline


# class LocalLLM:
#     def __init__(self):
#         self.generator = pipeline(
#             "text-generation",
#             model="microsoft/phi-2",
#             max_new_tokens=10
#         )

#     def generate(self, prompt):
#         print("Prompt : ", prompt)
#         result = self.generator(prompt)

#         print("Generated text : ",result)
#         return result[0]["generated_text"]

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


class LocalLLM:
    def __init__(self):
        model_name = "microsoft/phi-2"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Fix warning
        tokenizer.pad_token = tokenizer.eos_token

        self.generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.2,
            top_p=0.9
        )

    def generate(self, prompt):
        result = self.generator(prompt)[0]["generated_text"]
        answer = result[len(prompt):].strip()
        return answer
