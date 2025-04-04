from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

while True:

    user_input = input()

    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Generate response
    output_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)

    # Get length of original input
    input_length = input_ids.shape[-1]

    # Decode response
    response = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)

    print("Bot:", response)