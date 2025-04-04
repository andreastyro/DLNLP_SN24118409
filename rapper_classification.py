import os
import pandas as pd
import torch
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, DistilBertConfig
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load CSV files
root_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(root_dir, "Datasets")

train_df = pd.read_csv(os.path.join(save_dir, "rapper_train.csv"))
val_df = pd.read_csv(os.path.join(save_dir, "rapper_val.csv"))
test_df = pd.read_csv(os.path.join(save_dir, "rapper_test.csv"))

# Encode labels
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["rapper"])
val_df["label"] = label_encoder.transform(val_df["rapper"])
test_df["label"] = label_encoder.transform(test_df["rapper"])

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize(example):
    return tokenizer(example["background"], truncation=True, padding="max_length", max_length=80)

train_dataset = train_dataset.map(tokenize)
val_dataset = val_dataset.map(tokenize)
test_dataset = test_dataset.map(tokenize)

#Set format for PyTorch
train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")

# Dataloader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

num_labels = label_encoder.classes_.size

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

epochs = 25

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):

    train_loss = 0
    val_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # Add attention mask

        # Forward pass
        optimizer.zero_grad()

        y_pred = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = y_pred.logits

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total
    avg_loss = train_loss / len(train_loader)
    train_losses.append(avg_loss)
    train_accuracies.append(train_accuracy)

    print(f"[{epoch+1}/{epochs}, Training Loss: {avg_loss}, Training Accuracy: {train_accuracy}")

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch: {epoch+1}"):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            y_pred = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = y_pred.logits

            loss = criterion(logits, labels)

            val_loss += loss
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_accuarcy = correct / total
    avg_loss = val_loss / len(val_loader)
    val_losses.append(avg_loss)
    val_accuracies.append(val_accuarcy)

    print(f"[{epoch+1}/{epochs}, Validation Loss: {avg_loss}, Validation Accuracy: {val_accuarcy}")

model_dir = os.path.join(root_dir, "rapper_classification_model")
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
