import torch

import os
import nltk
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import word_tokenize

# === Setup ===
model_path = "lyrics_generator_model"  # your pre-trained model dir
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
ref_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
ref_model.eval()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# === Simple prompts ===
rappers = ["Drake", "Eminem", "Jay-Z", "Kanye West"]
prompts = [f"<RAPPER>: {r}\n<LYRICS>:" for r in rappers]

def reward_fn(text):
    tokens = word_tokenize(text.lower())
    bigrams = list(nltk.bigrams(tokens))
    unique_bigrams = set(bigrams)
    if len(bigrams) == 0:
        return 0.0
    return len(unique_bigrams) / len(bigrams)  # diversity score between 0 and 1

# === Simple reward: bigram diversity ===
def ppo_step(prompt):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    # Generate response from both models
    with torch.no_grad():
        ref_output = ref_model.generate(**inputs, max_new_tokens=50, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    output = model.generate(**inputs, max_new_tokens=50, do_sample=True, pad_token_id=tokenizer.eos_token_id)

    # Decode response and compute reward
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    reward = reward_fn(response)

    # Re-tokenize the generated response
    response_inputs = tokenizer(response, return_tensors="pt", padding=True).to(device)
    labels = response_inputs["input_ids"]

    # Compute log probs / losses from both models
    log_probs = model(**response_inputs, labels=labels).loss
    ref_log_probs = ref_model(**response_inputs, labels=labels).loss

    # Compute KL and final loss
    kl = log_probs - ref_log_probs
    loss = -reward + kl

    # Backprop
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return prompt, response, reward

# === Training loop ===
print("Training with PPO (manual, minimal)...")
for step in range(100):
    prompt = random.choice(prompts)
    prompt, response, reward = ppo_step(prompt)

    if step % 10 == 0:
        print(f"\nStep {step} — Reward: {reward:.4f}")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")

# === Save model ===
save_path = "lyrics_generator_model_ppo_simple"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n Saved model to {save_path}")
