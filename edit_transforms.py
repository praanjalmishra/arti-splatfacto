import json
import random

# Parameters (you can modify these)
input_json_path = "data/gs_t/transforms_post.json"
output_json_path = "data/gs_t/transforms_finetune.json"
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
seed = 42  # for reproducibility

# Load the original JSON
with open(input_json_path, 'r') as f:
    data = json.load(f)

# Shuffle frames for random splitting
frames = data['frames']
random.seed(seed)
random.shuffle(frames)


# Split frames
total = len(frames)
train_end = int(train_ratio * total)
val_end = train_end + int(val_ratio * total)

train_frames = frames[:train_end]
val_frames = frames[train_end:val_end]
test_frames = frames[val_end:]

# Extract filenames
def extract_file_paths(frames):
    return [frame['file_path'] for frame in frames]

data['train_filenames'] = extract_file_paths(train_frames)
data['val_filenames'] = extract_file_paths(val_frames)
data['test_filenames'] = extract_file_paths(test_frames)

# Write the updated JSON
with open(output_json_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Updated JSON saved to {output_json_path}")
