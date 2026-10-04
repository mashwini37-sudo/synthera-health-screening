import os
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "dataset_final")

rows = []

for label in ["normal", "anemic"]:
    folder = os.path.join(BASE, label)

    print("Checking folder:", folder)

    if not os.path.exists(folder):
        print("Folder not found:", folder)
        continue

    for f in os.listdir(folder):
        path = os.path.join(folder, f)

        if not os.path.isfile(path):
            continue

        try:
            Image.open(path).convert("RGB")
            rows.append({"path": path, "label": label})
        except Exception:
            print("Bad image skipped:", f)

df = pd.DataFrame(rows)

if len(df) == 0:
    print("No images found!")
    exit()

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

split_index = int(0.8 * len(df))

df["split"] = "val"
df.loc[:split_index - 1, "split"] = "train"

output_file = os.path.join(HERE, "phase1_index.csv")
df.to_csv(output_file, index=False)

print()
print("========== PHASE 1 RESULT ==========")
print("Total images:", len(df))
print()
print("Train / Validation:")
print(df["split"].value_counts())
print()
print("Normal / Anemic:")
print(df["label"].value_counts())
print()
print("Split + Label:")
print(df.groupby(["split", "label"]).size())
print()
print("CSV created successfully:")
print(output_file)
