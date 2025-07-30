import torch

# Load the .pt file
file_path = "data/gs_t/obj3Dseg0_updated.pt"
data = torch.load("data/gs_t/obj3Dseg0_updated.pt", map_location="cpu")


# update the .pt file

# Define your values (converted to torch tensors)
joint_axis = torch.tensor([4.23537969e-04, 7.79243400e-03, 9.99969549e-01], dtype=torch.float32)
joint_pivot = torch.tensor([-1.08500869, -0.07106712, 0.00750027], dtype=torch.float32)
joint_angle = torch.tensor(-1.2286, dtype=torch.float32)

# Add to the dictionary under explicit keys
data["joint_axis"] = joint_axis
data["joint_pivot"] = joint_pivot
data["joint_angle"] = joint_angle

# Optional: Group into one sub-dictionary instead
# data["joint_params"] = {
#     "axis": joint_axis,
#     "pivot": joint_pivot,
#     "angle": joint_angle
# }

# Save it back
torch.save(data, file_path)
print(f"[✅] Added joint_axis, joint_pivot, and joint_angle to {file_path}")

# Print top-level structure
print(f"Top-level type: {type(data)}")

if isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))
    
    # Print info for each key
    for key, value in data.items():
        print(f"\nKey: {key}")
        print(f"  Type: {type(value)}")
        
        # Print a summary of content if possible
        if isinstance(value, torch.Tensor):
            print(f"  Shape: {value.shape}, Dtype: {value.dtype}")
        elif isinstance(value, dict):
            print(f"  Nested keys: {list(value.keys())}")
        else:
            print(f"  Value: {value}")
else:
    print("Loaded object is not a dictionary. Type:", type(data))



