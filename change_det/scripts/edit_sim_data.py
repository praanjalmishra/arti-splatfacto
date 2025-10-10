# Edit transforms.json file in the NeRF dataset with scene changes
import json
import os
import re


def eval_new(json_file: str, output_file: str):
    """Evaluate on new views instead of old views

    Args:
        json_file: Input transforms.json file.
        output_file: Path to the output json file.
    """
    # load json files
    with open(json_file) as f1:
        data = json.load(f1)

    # Check if rgb_new is already in the val_filenames, if so, raise an error
    for fname in data['val_filenames']:
        assert 'rgb_new' not in fname, "'rgb_new' already in filename"
    for fname in data['test_filenames']:
        assert 'rgb_new' not in fname, "'rgb_new' already in filename"

    # Replace 'rgb' with 'rgb_new' to evaluate on new views
    data['val_filenames'] = [
        fname.replace('rgb', 'rgb_new')
        for fname in data['val_filenames']
    ]
    data['test_filenames'] = [
        fname.replace('rgb', 'rgb_new')
        for fname in data['test_filenames']
    ]

    # Write the updated data to the output file
    with open(output_file, 'w') as fout:
        json.dump(data, fout, indent=4)


def eval_old(json_file: str, output_file: str):
    """
    Evaluate on old views instead of new views.

    Args:
        json_file: Input transforms.json file.
        output_file: Path to the output json file.
    """
    with open(json_file) as f1:
        data = json.load(f1)

    # Check if rgb_new is already in the val_filenames, if so, raise an error
    for fname in data['val_filenames']:
        assert 'rgb_new' in fname, "Already eval on old views!"
    for fname in data['test_filenames']:
        assert 'rgb_new' in fname, "Already eval on old views!"

    # Replace 'rgb' with 'rgb_new' to evaluate on new views
    data['val_filenames'] = [
        fname.replace('rgb_new', 'rgb')
        for fname in data['val_filenames']
    ]
    data['test_filenames'] = [
        fname.replace('rgb_new', 'rgb')
        for fname in data['test_filenames']
    ]

    # Write the updated data to the output file
    with open(output_file, 'w') as fout:
        json.dump(data, fout, indent=4)


def eval_on_all_pix(json_file: str, output_file: str):
    """
    Evaluate on all pixels instead of masked pixels.

    Args:
        json_file: Input transforms.json file.
        output_file: Path to the output json file.
    """
    with open(json_file) as f1:
        data = json.load(f1)

    if eval_on_all_pix:
        for fname in data['val_filenames']:
            for frame in data['frames']:
                if frame['file_path'] == fname:
                    del frame['mask_path']
    
    # Write the updated data to the output file
    with open(output_file, 'w') as fout:
        json.dump(data, fout, indent=4)


def remove_mask_path(json_file, output_file):
    """
    Remove the mask paths from transforms.json.

    Args:
        json_file: Input transforms.json file.
        output_file: Path to the output json file.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)

    for frame in data['frames']:
        frame.pop('mask_path', None)

    # Write the updated JSON data back to the new file
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)


def extract_last_number(s):
    # finds the last group of digits in the string
    match = re.search(r'(\d+)(?!.*\d)', s)  
    if match:
        return int(match.group())
    else:
        raise ValueError(f"No number found in string {s}")


def add_mask_path(json_file, output_file):
    """
    Add the mask paths to transforms.json.

    Args:
        json_file: Input transforms.json file.
        output_file: Path to the output json file.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)

    has_mask_path = True
    for frame in data['frames']:
        if 'mask_path' not in frame:
            has_mask_path = False
            file_id = extract_last_number(frame['file_path'])
            if 'rgb_new' in frame['file_path']:
                mask_folder = 'masks_new'
            elif "rgb" in frame["file_path"]:
                mask_folder = 'masks'
            else:
                raise ValueError(
                    f"Expect rgb img filename to be rgb/frame_xxxxx.png " +
                    f" but got {frame['file_path']}"
                )
            mask_filename = f'mask_{file_id:05d}.png'
            frame['mask_path'] = os.path.join(mask_folder, mask_filename)
    
    if has_mask_path:
        print("Warning: All frames already have a mask_path.")

    # Write the updated JSON data back to the new file
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)


def add_depth_path(json_file, output_file):
    """
    Add the depth paths to transforms.json.

    Args:
        json_file: Input transforms.json file.
        output_file: Path to the output json file.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)

    has_depth_path = True
    for frame in data['frames']:
        if 'depth_file_path' not in frame:
            has_depth_path = False
            file_id = extract_last_number(frame['file_path'])
            if 'rgb_new' in frame['file_path']:
                depth_folder = 'depths_new'
            elif "rgb" in frame["file_path"]:
                depth_folder = 'depths'
            else:
                raise ValueError(
                    f"Expect rgb img filename to be rgb/frame_xxxxx.png " +
                    f" but got {frame['file_path']}"
                )
            depth_filename = f'depth_{file_id:05g}.png'
            frame['depth_file_path'] = os.path.join(depth_folder, depth_filename)
    
    if has_depth_path:
        print("Warning: All frames already have a mask_path.")

    # Write the updated JSON data back to the new file
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)


def remove_depth_path(json_file, output_file):
    """
    Remove the depth paths from transforms.json.

    Args:
        json_file: Input transforms.json file.
        output_file: Path to the output json file.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)

    for frame in data['frames']:
        frame.pop('depth_file_path', None)

    # Write the updated JSON data back to the new file
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)


def remove_pretrain_views(json_file, output_file):
    """
    Exclude pre-training views from the training data

    Args:
        json_file: Input transforms.json file.
        output_file: Path to the output json file.  
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    assert 'train_filenames' in data,\
        "No training data found in the json file."

    # Remove pre-training views from the training data
    train_filenames = []
    for filename in data['train_filenames']:
        if "rgb_new" in filename:
            train_filenames.append(filename)
    data['train_filenames'] = train_filenames
    frames = []
    for fr in data['frames']:
        if "rgb_new" in fr['file_path']:
            frames.append(fr)
    data['frames'] = frames

    # Write the updated JSON data back to the output file
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)


def change_new_view_indices(json_file, output_file, new_view_indices):
    """
    Change the indices of new views in transforms.json.

    Args:
        json_file: Input transforms.json file.
        output_file: Path to the output json file.
        new_view_indices: List of indices of new views.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)

    frames = data['frames']
    eval_filenames = data['val_filenames']
    train_filenames = data['train_filenames']

    # First flip rgb_new filenames in train_filenames back to rgb
    for filename in train_filenames:
        if 'rgb_new' in filename:
            new_filename = filename.replace('rgb_new', 'rgb')
            if new_filename in eval_filenames:
                raise ValueError(
                    f"Cannot change {filename} to {new_filename} " +
                    "since it is in eval_filenames"
                )
            train_filenames[train_filenames.index(filename)] = new_filename

    # Identify the filenames for rgb_new in the frames list using the indices
    rgb_new_filenames = [
        frames[i]['file_path'] for i in range(len(frames))
        if 'rgb_new' in frames[i]['file_path']
    ]

    # Get the subset of rgb_new_filenames using the new_view_indices
    subset_rgb_new_filenames = [rgb_new_filenames[i] for i in new_view_indices]

    # Use rgb_new files to replace rgb files in train_filenames
    for filename in subset_rgb_new_filenames:
        old_filename = filename.replace('rgb_new', 'rgb')        
        if filename in eval_filenames:
            raise ValueError(
                f'Cannot change {old_filename} to {filename} ' +
                'since it is in eval_filenames'
            )
        if not old_filename in data['train_filenames']:
            # no matching old view found
            # replace the next available old view instead
            # wrap around if reached the end of the list
            new_view_id = extract_last_number(filename)
            while old_filename not in data['train_filenames']:
                new_view_id = (new_view_id + 1) % len(data['train_filenames'])
                # replace the next available old view with the new view
                old_filename = f"rgb/frame_{new_view_id:05d}.png"         
            print(
                f"Warning: {filename} in old views is used for evaluation. "
                f"Replacing {old_filename} with new view instead."
            )
        train_filenames[train_filenames.index(old_filename)] = filename

    # Write the updated JSON data back to the output file
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--json', "-i", type=str, required=True)
    parser.add_argument('--out', "-o", type=str, required=True)
    parser.add_argument('--remove_masks', "-rm", action='store_true')
    parser.add_argument('--add_masks', "-am", action='store_true')
    parser.add_argument('--add_depth', "-ad", action='store_true')
    parser.add_argument('--remove_depth', "-rd", action='store_true')
    parser.add_argument('--remove_pretrain', "-rp", action='store_true')
    parser.add_argument("--eval_new", "-en", action='store_true')
    parser.add_argument("--eval_old", "-eo", action='store_true')
    parser.add_argument("--eval_all_pix", "-ea", action='store_true')
    parser.add_argument(
        "--new_view_indices", "-ni", type=int, nargs='+', default=[]
    )

    args = parser.parse_args()

    if args.remove_masks:
        assert not args.add_masks, \
            "Cannot remove and add masks at the same time."
        remove_mask_path(args.json, args.out)
    
    if args.add_masks:
        add_mask_path(args.json, args.out)

    if args.add_depth:
        add_depth_path(args.json, args.out)

    if args.remove_depth:
        remove_depth_path(args.json, args.out)
    
    if args.remove_pretrain:
        remove_pretrain_views(args.json, args.out)

    if args.eval_new:
        assert not args.eval_old, \
            "Cannot eval new and old views at the same time."
        eval_new(args.json, args.out)
    
    if args.eval_old:
        eval_old(args.json, args.out)

    if args.eval_all_pix:
        eval_on_all_pix(args.json, args.out)

    if args.new_view_indices:
        change_new_view_indices(args.json, args.out, args.new_view_indices)