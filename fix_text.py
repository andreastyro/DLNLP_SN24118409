import os
import pandas as pd
from ftfy import fix_text
import spacy

root_dir = os.path.dirname(os.path.abspath(__file__))
dataset = os.path.join(root_dir, "rapper_lyrics.csv")
save_dir = os.path.join(root_dir, "rapper_lyrics.csv")

# Load your CSV file
df = pd.read_csv(dataset)

# Load model and dataset
nlp = spacy.load("en_core_web_sm")
"""
# Redact named entities
def redact(text):
    doc = nlp(text)
    for ent in reversed(doc.ents):
        if ent.label_ in ["PERSON"]:
            text = text[:ent.start_char] + f"[{ent.label_}]" + text[ent.end_char:]
    return text
"""

#df["background"] = df["background"].apply(redact)
#df.to_csv(no_entities, index=False)

# Fix text encoding issue
df["current_lyric"] = df["current_lyric"].apply(fix_text)
df["next_lyric"] = df["next_lyric"].apply(fix_text)
df["song_title"] = df["song_title"].apply(fix_text)
df["rapper_name"] = df["rapper_name"].apply(fix_text)

# Save the cleaned file
df.to_csv(save_dir, index=False, encoding="utf-8")

print("Encoding fixed! Saved as 'rapper_backgrounds_clean.csv'")
