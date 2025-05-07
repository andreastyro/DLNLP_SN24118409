import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Load trained model ===
model_path = "lyrics_generator_model_ppo_simple"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# === Generation function ===
def generate_lyrics(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# === Generate 10 Kanye West lyrics ===
prompt_base = "<RAPPER>: Kanye West\n<LYRICS>:"
print(" Kanye West Lyrics:\n")

for i in range(10):
    lyric = generate_lyrics(prompt_base)
    print(f"--- Lyric {i+1} ---\n{lyric}\n")
