import os
import math
import pandas as pd
import torch
import pickle
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import nltk; nltk.download('punkt', quiet=True)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")   # fast 384-d encoder

def distinct_n(corpus, n):
    total, uniq = 0, set()
    for text in corpus:
        tok = nltk.word_tokenize(text.lower())
        for i in range(len(tok) - n + 1):
            total += 1
            uniq.add(tuple(tok[i : i + n]))
    return len(uniq) / max(total, 1)


tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
special_tokens = {'additional_special_tokens': ['<RAPPER>', '<LYRICS>']}
tokenizer.add_special_tokens(special_tokens)

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

root_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(root_dir, "Datasets")

train_df = pd.read_csv(os.path.join(save_dir, "lyrics_train.csv"))
val_df = pd.read_csv(os.path.join(save_dir, "lyrics_val.csv"))
test_df = pd.read_csv(os.path.join(save_dir, "lyrics_test.csv"))

# Combine rapper name and lyrics into a single prompt-style string
def format_prompt(row):
    return f"<RAPPER>: {row['rapper']}\n<LYRICS>: {row['lyric']}"

train_df["text"] = train_df.apply(format_prompt, axis=1)
val_df["text"] = val_df.apply(format_prompt, axis=1)
test_df["text"] = test_df.apply(format_prompt, axis=1)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[["text"]])
val_dataset = Dataset.from_pandas(val_df[["text"]])
test_dataset = Dataset.from_pandas(test_df[["text"]])

print(train_dataset[0])

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=100)

train_dataset = train_dataset.map(tokenize)
val_dataset = val_dataset.map(tokenize)
test_dataset = test_dataset.map(tokenize)

#Set format for PyTorch
train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

epochs = 3

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

generations = []
references = []

test_loss = 0



for epoch in range(epochs):
    
    train_loss = 0
    val_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch{epoch+1}"):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()

        optimizer.zero_grad()

        y_pred = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Dont forget exploding gradients

        loss = y_pred.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f} "f"| PPL: {math.exp(avg_train_loss):.2f}")

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch{epoch+1}"):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            y_pred = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


            loss = y_pred.loss
            val_loss += loss.item()
        
            gen_data = model.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=100,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        ) 
            
            for ref, gen in zip(labels, gen_data):
                references.append(tokenizer.decode(ref,  skip_special_tokens=True))
                generations.append(tokenizer.decode(gen, skip_special_tokens=True))


        avg_val_loss = val_loss / len(val_loader)
        val_ppl      = math.exp(avg_val_loss)

        # embedding-similarity (cosine of paired lines)
        emb_gen = embed_model.encode(generations, convert_to_tensor=True)
        emb_ref = embed_model.encode(references,  convert_to_tensor=True)
        emb_sim = util.cos_sim(emb_gen, emb_ref).diagonal().mean().item()

        # lexical diversity
        d1 = distinct_n(generations, 1)
        d2 = distinct_n(generations, 2)

        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f} "
              f"| PPL: {val_ppl:.2f} | Emb-Sim: {emb_sim:.3f} "
              f"| Dist-1: {d1:.3f} | Dist-2: {d2:.3f}")


with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()

        y_pred = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = y_pred.loss
        test_loss += loss.item()
    
        gen_ids = model.generate(input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=40,
                                do_sample=True, top_p=0.95, temperature=0.9)
        generations += [tokenizer.decode(g, skip_special_tokens=True)
                        for g in gen_ids]

    avg_test_loss = test_loss / len(test_loader)

print(f"Testing Loss: {avg_test_loss:.4f} | PPL: {math.exp(avg_test_loss):.2f} "
      f"| Dist-1: {distinct_n(generations,1):.3f} "
      f"| Dist-2: {distinct_n(generations,2):.3f}")
# Save model
save_path = os.path.join(root_dir, "lyrics_generator_model")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("Model and tokenizer saved.")