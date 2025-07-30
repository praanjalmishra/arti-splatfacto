import torch

# Load the .pt file
data = torch.load("data/gs_t/obj3Dseg0_updated.pt", map_location="cpu")

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
