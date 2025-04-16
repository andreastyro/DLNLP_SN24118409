import os
import pandas as pd
import torch
import pickle
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

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
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=64)

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

epochs = 10

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

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

        loss = y_pred.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch{epoch+1}"):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            y_pred = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = y_pred.loss
            val_loss = loss.item()
        
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

with torch.no_grad:
    for batch in tqdm(test_loader, desc="Testing"):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids.clone()

        y_pred = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = y_pred.loss
        test_loss = loss.item()
    
    avg_test_loss = test_loss / len(test_loader)

    print(f"Epoch {epoch+1} Testing Loss: {avg_test_loss:.4f}")

# Save model
save_path = os.path.join(root_dir, "gpt2_rapper_generator")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("Model and tokenizer saved.")