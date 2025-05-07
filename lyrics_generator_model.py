import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, AutoModelForSequenceClassification

root_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(root_dir, "lyrics_generator_model")

max_new_tokens = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

llm = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "<RAPPER>: Drake\n<LYRICS>:"

inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

K = 20
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

for i, text in enumerate(decoded_outputs):
    print(f"\n Option {i+1}:\n{text}\n")


"""
### Load reward models
names_rmodel = [
    "Ray2333/gpt2-large-harmless-reward_model",
    "Ray2333/gpt2-large-helpful-reward_model"
]

rmodels = list()
rtokenizers = list()
for name_rmodel in names_rmodel:
    rmodels.append(
        AutoModelForSequenceClassification.from_pretrained(
            name_rmodel,
            num_labels=1,
        ).to(device)
    )
    rtokenizers.append(
        AutoTokenizer.from_pretrained(name_rmodel)
    )
    rtokenizers[-1].pad_token = rtokenizers[-1].eos_token
    rmodels[-1].config.pad_token_id = rtokenizers[-1].pad_token_id
    rmodels[-1].eval()

def evaluate_rewards(
    texts,
    rtokenizers,
    rmodels,
    device,
    batch_size=4,
):
    assert len(rtokenizers) == len(rmodels), f"rtokenizers ({len(rtokenizers)}) and rmodels ({len(rmodels)}) need to have same length"
    batch_size = min(batch_size, len(texts))
    res = list()
    for i in range(len(rtokenizers)):
        reward_tokenizer = rtokenizers[i]
        rmodel = rmodels[i]
        # Tokenize pairs for reward model
        encoded_texts = reward_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        encoded_texts = {k: v.to(device) for k, v in encoded_texts.items()}

        # Calculate rewards
        with torch.no_grad():
            idx_start = 0
            reward_outputs = list()
            for i in range(len(texts) // batch_size):
                idx_end = idx_start + batch_size
                batch_encoded_texts = {
                    k: v[idx_start:idx_end].to(device) for k, v in encoded_texts.items()
                }
                reward_outputs.append(
                    rmodel(**batch_encoded_texts).logits
                )
                idx_start += batch_size
            reward_outputs = torch.cat(reward_outputs, axis=0)
            res.append(reward_outputs)

    res = torch.cat(res, axis=1)
    # print(f"[evaluate_rewards] len(texts) {len(texts)} | len(res) {len(res)}") # Use when needed

    return res.to(torch.float16).detach().cpu().numpy()

decoded_outputs = tokenizer.batch_decode(outputs)
evaluations = evaluate_rewards(
    decoded_outputs,
    rtokenizers,
    rmodels,
    device,
    batch_size=4
)

for i, lyric in enumerate(decoded_outputs):
    print(f"\nOption {i+1} (Score: {evaluations[i][0]:.4f}):\n{lyric.replace(prompt, '').replace('<|endoftext|>', '').strip()}")

idx_bestofk_reward0 = evaluations[:, 0].argmax(axis=0)
decoded_outputs[idx_bestofk_reward0][len(prompt):]
print(decoded_outputs)

"""