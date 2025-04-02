import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from collections import Counter

root_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(root_dir, "Plots")
os.makedirs(plots_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CLINC150 dataset (use the 'small' version to keep things light)
dataset = load_dataset("clinc_oos", "small")

# Load a pretrained DistilBERT tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=32)

tokenized_dataset = dataset.map(preprocess)
tokenized_dataset = tokenized_dataset.rename_column("intent", "labels")

#print("Before", tokenized_dataset['train'][0])

tokenized_dataset.set_format("torch", columns = ["input_ids", "attention_mask", "labels"])

#print("After", tokenized_dataset['train'][0])

train_loader = DataLoader(tokenized_dataset["train"], batch_size=16, shuffle=True)
val_loader = DataLoader(tokenized_dataset["validation"], batch_size=32)

# Load the DistilBERT model for classification - Faster than BERT
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=151)
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters() ,lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

epochs = 10
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
# Training loop

predictions_dict = {}

for epoch in range(epochs):
    correct = 0
    train_loss = 0
    val_loss = 0
    total_train = 0
    total_val = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        # Loss
        loss = loss_fn(logits, labels)

        # Backward propagation        
        loss.backward()
        optimizer.step()

        # Total Loss
        train_loss += loss
        predictions = torch.argmax(logits, dim=1)

        correct += (predictions == labels).sum().item()
        total_train += labels.size(0)
        

    train_accuracy = correct / total_train
    train_losses.append((train_loss / len(train_loader)).item())
    train_accuracies.append(train_accuracy)

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}')

    correct = 0

    with torch.no_grad():

        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            # Loss
            loss = loss_fn(logits, labels)

            # Total Loss
            val_loss += loss
            predictions = torch.argmax(logits, dim=1) 

            correct += (predictions == labels).sum().item()
            total_val += labels.size(0)

            #predictions = predictions.tolist()
            #predictions_dict = Counter(predictions)


    val_accuracy = correct / total_val
    val_losses.append((val_loss / len(val_loader)).item())
    val_accuracies.append(val_accuracy)
    #print(predictions_dict)
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.4f}')

epoch_range = list(range(1, epochs + 1))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(epoch_range, train_losses, label="Training Loss")
axes[0].plot(epoch_range, val_losses, label="Validation Loss")
axes[0].set_title("Loss vs Epochs")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(epoch_range, train_accuracies, label="Training Accuracu")
axes[1].plot(epoch_range, val_accuracies, label="Validation Accuracy")
axes[1].set_title("Accuracy vs Epochs")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

fig.savefig(os.path.join(plots_dir, "Figure"))