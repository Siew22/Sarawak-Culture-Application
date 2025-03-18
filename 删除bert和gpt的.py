import shutil
import os

folders_to_delete = [
    "C:\Users\User\AppData\Local\Programs\Python\Python310\ai_travel_assistant\models/bert_classifier",
    "C:\Users\User\AppData\Local\Programs\Python\Python310\ai_travel_assistant\models/gpt2_trained"
]

for folder in folders_to_delete:
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
        print(f"Deleted folder: {folder}")
    else:
        print(f"Folder not found: {folder}")
