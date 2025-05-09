def demo():
    import os
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import pandas as pd

    root_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root_dir, "lyrics_generator_model")

    max_new_tokens = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    llm = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"  # Important for decoder-only models
    tokenizer.pad_token = tokenizer.eos_token

    rappers = ["2pac", "50 Cent", "Drake", "Eminem", "Jay-Z", "Kendrick Lamar", "Nas", "Pusha T", "Travis Scott"]

    K = 4
    all_lyrics = []

    for rapper in rappers:
        prompt = f"<RAPPER>: {rapper}\n<LYRICS>:"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        outputs = llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_return_sequences=K,
            do_sample=True,
            temperature=0.9,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        decoded_outputs = [
            tokenizer.decode(output, skip_special_tokens=True).replace("<|endoftext|>", "").strip()
            for output in outputs
        ]

        print (decoded_outputs)

if __name__ == "__main__":
    demo()