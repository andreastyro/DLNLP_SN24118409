import os
import pandas as pd
from ftfy import fix_text
import spacy

root_dir = os.path.dirname(os.path.abspath(__file__))
dataset = os.path.join(root_dir, "rapper_biographies.csv")
save_dir = os.path.join(root_dir, "dataset_clean.csv")

# Load your CSV file
df = pd.read_csv(dataset)

# Load model and dataset
nlp = spacy.load("en_core_web_sm")

# Redact named entities
def redact(text):
    doc = nlp(text)
    for ent in reversed(doc.ents):
        if ent.label_ in ["PERSON"]:
            text = text[:ent.start_char] + f"[{ent.label_}]" + text[ent.end_char:]
    return text

df["background"] = df["background"].apply(redact)
#df.to_csv(no_entities, index=False)


# Fix text encoding issues
df["background"] = df["background"].apply(fix_text)

# Save the cleaned file
df.to_csv(save_dir, index=False, encoding="utf-8")

print("Encoding fixed! Saved as 'rapper_backgrounds_clean.csv'")
