import json

with open("data/gs_t/transforms_finetune.json") as f:
    data = json.load(f)

frame_paths = {frame["file_path"] for frame in data["frames"]}
val_paths = set(data["val_filenames"])

missing = val_paths - frame_paths
if missing:
    print("❌ These val_filenames are not in frames:", missing)
else:
    print("✅ All val_filenames exist in frames.")
