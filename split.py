import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ftfy import fix_text

root_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(root_dir, "Datasets")
os.makedirs(save_dir, exist_ok=True)

dataset = os.path.join(root_dir, "rapper_lyrics_dataset.csv")

# Load the cleaned dataset
df = pd.read_csv(dataset)

#df["background"] = df["background"].apply(fix_text)

train_df = pd.DataFrame()
temp_df = pd.DataFrame()

#filtered_df = df[df["rapper"].isin(["Eminem", "Nas"])]

for rapper in df["rapper"].unique():
    rapper_data = df[df["rapper"] == rapper]
    train, temp = train_test_split(rapper_data, test_size=0.3, random_state=42, shuffle=True)

    train_df = pd.concat([train_df, train]) # Concatenates each rapper's bios each iteration
    temp_df = pd.concat([temp_df, temp])

# Now split the remaining 30% temp set into val and test (50/50)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

train_df.to_csv(os.path.join(save_dir, "lyrics_train.csv"), index=False)
val_df.to_csv(os.path.join(save_dir, "lyrics_val.csv"), index=False)
test_df.to_csv(os.path.join(save_dir, "lyrics_test.csv"), index=False)
