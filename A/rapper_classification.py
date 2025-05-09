def rapper_classifier():

    import os
    import pandas as pd
    import torch
    import pickle
    from datasets import Dataset
    from transformers import BertTokenizerFast, BertForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    # Load CSV files
    root_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(root_dir, "Datasets")

    train_df = pd.read_csv(os.path.join(save_dir, "lyrics_train.csv"))
    val_df = pd.read_csv(os.path.join(save_dir, "lyrics_val.csv"))
    test_df = pd.read_csv(os.path.join(save_dir, "lyrics_test.csv"))

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
        return tokenizer(example["lyrics"], truncation=True, padding="max_length", max_length=80)

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

    num_labels = len(label_encoder.classes_)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = 2

    train_losses = []
    train_accuracies = []
    train_f1_scores = [] 
    val_losses = []
    val_accuracies = []
    val_f1_scores = []    

    for epoch in range(epochs):

        train_loss = 0
        correct = 0
        total = 0
        all_train_preds = []  
        all_train_labels = [] 

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            optimizer.zero_grad()

            y_pred = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = y_pred.logits

            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Collect predictions and labels for F1 score
            all_train_preds.extend(predictions.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())

        train_accuracy = correct / total
        train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')  # Calculate training F1
        avg_loss = train_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        train_f1_scores.append(train_f1)

        print(f"[{epoch+1}/{epochs}] Training Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")

        # Validation
        val_loss = 0
        correct = 0
        total = 0
        all_val_preds = []
        all_val_labels = []

        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch: {epoch+1}"):

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                y_pred = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = y_pred.logits

                loss = criterion(logits, labels)

                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # Collect predictions and labels for F1 score
                all_val_preds.extend(predictions.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        val_accuracy = correct / total
        val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        avg_loss = val_loss / len(val_loader)
        
        val_losses.append(avg_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)

        print(f"[{epoch+1}/{epochs}] Validation Loss: {avg_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")

    # Testing
    correct = 0
    test_loss = 0 
    total = 0 
    all_test_preds = []
    all_test_labels = []

    model.eval() 
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing"):
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            y_pred = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = y_pred.logits

            loss = criterion(logits, labels)

            test_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Collect predictions and labels for F1 score
            all_test_preds.extend(predictions.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    test_accuracy = correct / total
    test_f1 = f1_score(all_test_labels, all_test_preds, average='macro')
    avg_loss = test_loss / len(test_loader)

    print(f"Testing Loss: {avg_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")

    # Save model artifacts
    model_dir = os.path.join(root_dir, "rapper_classification_model_for_lyrics")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    with open(os.path.join(model_dir, "label_encoder_for_lyrics.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)

if __name__ == "__main__":
    rapper_classifier()