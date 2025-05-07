import os
import pandas as pd
import torch
import pickle
from transformers import BertTokenizerFast, BertForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel

root_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(root_dir, "rapper_classification_model")
label_path = os.path.join(model_path, "label_encoder.pkl")

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)

# Load label encoder (using pickle)
with open(label_path, "rb") as f:
    label_encoder = pickle.load(f)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

while True:

    text = input("Enter rapper's biography: ")

    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=80, return_tensors='pt')

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        print(logits)
        print(predictions)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        # Print all rapper probabilities for debugging
        print("\nProbabilities for each rapper:")
        for i in range(len(label_encoder.classes_)):
            rapper = label_encoder.inverse_transform([i])[0]
            prob = probabilities[0][i].item() 
            print(f"{rapper}: {prob:.4f}")

        # Get prediction
        predictions = torch.argmax(logits, dim=1)
        print("\nRaw prediction tensor:", predictions)
        
        # Get the integer value from the tensor
        prediction_idx = predictions.item()  # Convert tensor to integer
        print("Prediction index:", prediction_idx)


    # Decode prediction using the integer value
    predicted_rapper = label_encoder.inverse_transform([prediction_idx])[0]
    print(f"\nPredicted rapper: {predicted_rapper}")