import torch
from typing import Dict, Any, Optional



class MultiJointStateManager:
    """Manages state loading/saving for multi-joint articulated Gaussians."""
    
    @staticmethod
    def save_state(model) -> Dict[str, Any]:
        """Save complete multi-joint state with per-joint metadata."""
        state = {}
        
        # Save all joint Gaussians
        for joint_id, params in model.all_gauss_params_obj.items():
            for name, param in params.items():
                state[f"joints.{joint_id}.gaussians.obj.{name}"] = param.data
        
        for joint_id, params in model.all_gauss_params_canon.items():
            for name, param in params.items():
                state[f"joints.{joint_id}.gaussians.canon.{name}"] = param.data
        
        # Save background
        for name, param in model.gauss_params_fixed.items():
            state[f"background.gaussians.{name}"] = param.data
        
        # Save joint parameters (pivot, axis, angles, etc.)
        for joint_id, params in model.all_joint_params.items():
            for name, param in params.items():
                state[f"joints.{joint_id}.params.{name}"] = param.data
        
        # === NEW: Save per-joint metadata ===
        joint_metadata_to_save = {}
        for joint_id in model.all_joint_params.keys():
            # Get metadata if it exists
            if hasattr(model, 'joint_metadata') and joint_id in model.joint_metadata:
                joint_metadata_to_save[joint_id] = model.joint_metadata[joint_id]
            else:
                # Infer from current state
                meta = {}
                
                # Infer num_frames from angles
                if "angles" in model.all_joint_params[joint_id]:
                    meta["num_frames"] = model.all_joint_params[joint_id]["angles"].shape[0]
                
                # Infer joint_type
                joint_type_attr = f'joint_type_{joint_id}'
                if hasattr(model, joint_type_attr):
                    meta["joint_type"] = getattr(model, joint_type_attr)
                elif hasattr(model, 'joint_type'):
                    meta["joint_type"] = model.joint_type
                
                # Infer limits from angles
                if "angles" in model.all_joint_params[joint_id]:
                    angles = model.all_joint_params[joint_id]["angles"]
                    meta["learned_limits"] = [angles.min().item(), angles.max().item()]
                
                joint_metadata_to_save[joint_id] = meta
        
        # Save global metadata
        state["_metadata"] = {
            "joint_ids": sorted(model.all_joint_params.keys()),
            "active_joint": model.config.active_joint_id,
            "training_mode": model.config.training_mode,
            "per_joint_metadata": joint_metadata_to_save,  # NEW!
        }
        
        return state
    
    @staticmethod
    def load_state(model, state_dict: Dict[str, Any], inference_mode: bool = False):
        """
        Smart loader that handles:
        1. Vanilla 3DGS checkpoints (needs partitioning)
        2. Single-joint checkpoints (legacy format)
        3. Multi-joint checkpoints (new format)
        """
        load_type = MultiJointStateManager._detect_checkpoint_type(state_dict)
        
        print(f"\n{'='*70}")
        print(f"[StateManager] Detected checkpoint type: {load_type}")
        print(f"[StateManager] Mode: {'INFERENCE' if inference_mode else 'TRAINING'}")
        print(f"[StateManager] Active joint: {model.config.active_joint_id}")
        print(f"{'='*70}\n")
        
        if load_type == "vanilla":
            # Fresh start: partition vanilla checkpoint
            MultiJointStateManager._load_vanilla(model, state_dict)
        
        elif load_type == "multi_joint":
            # Continue multi-joint training or render trained model
            if inference_mode:
                MultiJointStateManager._load_multi_joint_inference(model, state_dict)
            else:
                MultiJointStateManager._load_multi_joint_training(model, state_dict)
        
        elif load_type == "legacy_single":
            # Old format: convert to new format
            converted = MultiJointStateManager._convert_legacy_to_new(state_dict)
            MultiJointStateManager.load_state(model, converted, inference_mode)
        
        else:
            raise ValueError(f"Unknown checkpoint type: {load_type}")
        
        # Set active pointers
        model._set_active_pointers()
    
    @staticmethod
    def _detect_checkpoint_type(state_dict: Dict[str, Any]) -> str:
        """Detect what type of checkpoint we're loading."""
        keys = list(state_dict.keys())
        
        # Check for vanilla 3DGS
        if "gauss_params.means" in keys or "_model.gauss_params.means" in keys:
            has_multi = any(k.startswith("joints.") for k in keys)
            if not has_multi:
                return "vanilla"
        
        # Check for new multi-joint format
        if any(k.startswith("joints.") and ".gaussians." in k for k in keys):
            return "multi_joint"
        
        # Check for legacy format
        if any(k.startswith("all_gauss_params_obj.") for k in keys):
            return "legacy_single"
        
        return "unknown"
    
    @staticmethod
    def _load_vanilla(model, state_dict: Dict[str, Any]):
        """Load vanilla 3DGS and partition for active joint."""
        print(f"[Vanilla Load] Partitioning vanilla checkpoint for {model.config.active_joint_id}...")
        
        # Ensure keys are in the right format
        vanilla_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model.gauss_params."):
                # _model.gauss_params.means -> gauss_params.means
                clean_key = key.replace("_model.", "")
                vanilla_state[clean_key] = value
            elif key.startswith("gauss_params."):
                vanilla_state[key] = value
        
        if "gauss_params.means" in vanilla_state:
            n_gaussians = vanilla_state["gauss_params.means"].shape[0]
            print(f"  Found {n_gaussians:,} Gaussians to partition")
            
            # This will partition into active joint + background
            model._initialize_and_partition(vanilla_state)
        else:
            print("  ⚠️  No Gaussians found in vanilla checkpoint!")
    
    @staticmethod
    def _load_multi_joint_inference(model, state_dict: Dict[str, Any]):
        """Load ALL joints for rendering (no validation, no partitioning)."""
        print(f"[Inference Load] Loading all joints from checkpoint...")
        
        device = model.device
        loaded_joints = set()
        
        # Load all joint Gaussians (frozen)
        for key, tensor in state_dict.items():
            if key.startswith("joints.") and ".gaussians." in key:
                # joints.joint_0.gaussians.obj.means
                parts = key.split('.')
                joint_id = parts[1]
                gauss_type = parts[3]  # 'obj' or 'canon'
                param_name = parts[4]
                
                loaded_joints.add(joint_id)
                
                target = model.all_gauss_params_obj if gauss_type == "obj" else model.all_gauss_params_canon
                if joint_id not in target:
                    target[joint_id] = torch.nn.ParameterDict()
                
                target[joint_id][param_name] = torch.nn.Parameter(
                    tensor.to(device), requires_grad=False
                )
            
            elif key.startswith("background.gaussians."):
                # background.gaussians.means
                param_name = key.split('.')[-1]
                model.gauss_params_fixed[param_name] = torch.nn.Parameter(
                    tensor.to(device), requires_grad=False
                )
            
            elif key.startswith("joints.") and ".params." in key:
                # joints.joint_0.params.angles
                parts = key.split('.')
                joint_id = parts[1]
                param_name = parts[3]
                
                if joint_id not in model.all_joint_params:
                    model.all_joint_params[joint_id] = torch.nn.ParameterDict()
                
                model.all_joint_params[joint_id][param_name] = torch.nn.Parameter(
                    tensor.to(device), requires_grad=False
                )
                
                if param_name == "angles":
                    print(f"  ✓ {joint_id}: Loaded {len(tensor)} angles "
                          f"[{tensor.min():.3f}, {tensor.max():.3f}] rad")
        
        print(f"\n  ✓ Loaded {len(loaded_joints)} joints: {sorted(loaded_joints)}")
    
    @staticmethod
    def _load_multi_joint_training(model, state_dict: Dict[str, Any]):
        """Load for continued training (validate, potentially partition new joint)."""
        active_id = model.config.active_joint_id
        device = model.device
        
        print(f"[Training Load] Loading for {active_id} training...")
        
        # Check if we need to partition a new joint
        existing_joints = set()
        for key in state_dict.keys():
            if key.startswith("joints.") and ".gaussians." in key:
                joint_id = key.split('.')[1]
                existing_joints.add(joint_id)
        
        needs_partition = active_id not in existing_joints
        
        if needs_partition:
            print(f"  → {active_id} not found in checkpoint")
            print(f"  → Will partition {active_id} from remaining scene")
            print(f"  → Existing joints: {sorted(existing_joints)}")
            
            # === RECONSTRUCT FULL SCENE FOR PARTITIONING ===
            # This creates: background + all_previous_joints (but NOT active_id)
            full_scene = MultiJointStateManager._reconstruct_full_scene(
                state_dict, exclude_joint=active_id
            )
            
            print(f"  → Reconstructed scene: {full_scene['gauss_params.means'].shape[0]:,} Gaussians")
            
            # === LOAD ALL PREVIOUS JOINTS FIRST (Before Partitioning) ===
            # This is CRITICAL - previous joints must exist before partitioning
            for key, tensor in state_dict.items():
                if key.startswith("joints.") and ".gaussians." in key:
                    parts = key.split('.')
                    joint_id = parts[1]
                    gauss_type = parts[3]  # 'obj' or 'canon'
                    param_name = parts[4]
                    
                    # Only load previous joints (not active, since it doesn't exist yet)
                    if joint_id == active_id:
                        continue
                    
                    target = model.all_gauss_params_obj if gauss_type == "obj" else model.all_gauss_params_canon
                    if joint_id not in target:
                        target[joint_id] = torch.nn.ParameterDict()
                    
                    target[joint_id][param_name] = torch.nn.Parameter(
                        tensor.to(device), requires_grad=False  # Previous joints are frozen
                    )
            
            # Load previous joint parameters
            for key, tensor in state_dict.items():
                if key.startswith("joints.") and ".params." in key:
                    parts = key.split('.')
                    joint_id = parts[1]
                    param_name = parts[3]
                    
                    if joint_id == active_id:
                        continue
                    
                    if joint_id not in model.all_joint_params:
                        model.all_joint_params[joint_id] = torch.nn.ParameterDict()
                    
                    model.all_joint_params[joint_id][param_name] = torch.nn.Parameter(
                        tensor.to(device), requires_grad=False
                    )
            
            print(f"  ✓ Loaded {len(model.all_gauss_params_obj)} previous joints")
            
            # === NOW PARTITION THE NEW JOINT ===
            # This will:
            # 1. Query mask on the reconstructed scene (background + previous joints)
            # 2. Split into: new_joint + remaining_background
            # 3. Update background and create active_id Gaussians
            print(f"\n{'='*70}")
            print(f"PARTITIONING {active_id}")
            print(f"{'='*70}")
            model._initialize_and_partition(full_scene)
            
        else:
            # === ACTIVE JOINT EXISTS - NORMAL LOAD ===
            print(f"  → {active_id} found in checkpoint, loading normally")
            
            # Load all joints (active + previous)
            for key, tensor in state_dict.items():
                if key.startswith("joints.") and ".gaussians." in key:
                    parts = key.split('.')
                    joint_id = parts[1]
                    gauss_type = parts[3]
                    param_name = parts[4]
                    
                    target = model.all_gauss_params_obj if gauss_type == "obj" else model.all_gauss_params_canon
                    if joint_id not in target:
                        target[joint_id] = torch.nn.ParameterDict()
                    
                    # Active joint: trainable, others: frozen
                    requires_grad = (joint_id == active_id and model.training)
                    
                    target[joint_id][param_name] = torch.nn.Parameter(
                        tensor.to(device), requires_grad=requires_grad
                    )
            
            # Load background
            for key, tensor in state_dict.items():
                if key.startswith("background.gaussians."):
                    param_name = key.split('.')[-1]
                    model.gauss_params_fixed[param_name] = torch.nn.Parameter(
                        tensor.to(device), requires_grad=False
                    )
        
        # === LOAD JOINT PARAMETERS (Both Scenarios) ===
        MultiJointStateManager._load_joint_params_training(model, state_dict, active_id)

    @staticmethod
    def _load_joint_params_training(model, state_dict: Dict[str, Any], active_id: str):
        """Load joint parameters with per-joint frame validation."""
        device = model.device
        
        active_joint_meta = model.metadata.get(f"joint_angles_{active_id}", 
                                            model.metadata.get("joint_angles", []))
        expected_frames_active = len(active_joint_meta)
        
        # === NEW: Load per-joint metadata from checkpoint ===
        checkpoint_metadata = {}
        if "_metadata" in state_dict and "per_joint_metadata" in state_dict["_metadata"]:
            checkpoint_metadata = state_dict["_metadata"]["per_joint_metadata"]
        
        # Initialize model's joint_metadata if not exists
        if not hasattr(model, 'joint_metadata'):
            model.joint_metadata = {}
        
        for key, tensor in state_dict.items():
            if not key.startswith("joints.") or ".params." not in key:
                continue
            
            parts = key.split('.')
            joint_id = parts[1]
            param_name = parts[3]
            
            if joint_id not in model.all_joint_params:
                model.all_joint_params[joint_id] = torch.nn.ParameterDict()
            
            # === NEW: Handle frame-dependent params with per-joint validation ===
            if param_name in {"angles", "angle_deltas"}:
                checkpoint_frames = tensor.shape[0]
                
                if joint_id == active_id:
                    # ACTIVE JOINT: Must match current dataset
                    if expected_frames_active != checkpoint_frames:
                        print(f"  ⚠️  Skipping {joint_id}.{param_name}: "
                            f"frame mismatch ({checkpoint_frames} vs {expected_frames_active})")
                        print(f"      Active joint will use freshly initialized angles from metadata")
                        continue
                    else:
                        print(f"  ✓ {joint_id}.{param_name}: {checkpoint_frames} frames (matches dataset)")
                
                else:
                    # PREVIOUS JOINT: Load with its original frame count
                    # No validation needed - we trust the checkpoint
                    print(f"  ✓ {joint_id}.{param_name}: {checkpoint_frames} frames (from checkpoint)")
                    
                    # Store metadata for this joint
                    if joint_id not in model.joint_metadata:
                        model.joint_metadata[joint_id] = {}
                    model.joint_metadata[joint_id]["num_frames"] = checkpoint_frames
                    
                    # Load additional metadata if available
                    if joint_id in checkpoint_metadata:
                        model.joint_metadata[joint_id].update(checkpoint_metadata[joint_id])
            
            # Active joint: trainable, others: frozen
            requires_grad = (joint_id == active_id and model.training)
            
            model.all_joint_params[joint_id][param_name] = torch.nn.Parameter(
                tensor.to(device), requires_grad=requires_grad
            )
            
            if param_name == "angles" and joint_id != active_id:
                # Also register as a buffer for easy access
                buffer_name = f'joint_angles_{joint_id}'
                model.register_buffer(buffer_name, tensor.to(device))
    
    @staticmethod
    def _reconstruct_full_scene(state_dict: Dict[str, Any], exclude_joint: Optional[str]) -> Dict[str, Any]:
        """
        Reconstruct full scene for partitioning.
        
        Returns a state_dict with keys like:
            gauss_params.means
            gauss_params.scales
            etc.
        
        This concatenates: background + all_joints_except_exclude
        """
        GAUSS = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        combined = {f"gauss_params.{p}": [] for p in GAUSS}
        
        print(f"\n  [Reconstruct] Building full scene (excluding {exclude_joint})...")
        
        # === 1. Add Background First ===
        bg_count = 0
        for key, tensor in state_dict.items():
            if key.startswith("background.gaussians."):
                param_name = key.split('.')[-1]
                combined[f"gauss_params.{param_name}"].append(tensor)
                if param_name == "means":
                    bg_count = tensor.shape[0]
        
        if bg_count > 0:
            print(f"  [Reconstruct]   Background: {bg_count:,} Gaussians")
        
        # === 2. Add All Previous Joints (Except Exclude) ===
        joint_counts = {}
        for key, tensor in state_dict.items():
            if key.startswith("joints.") and ".gaussians.obj." in key:
                # joints.joint_0.gaussians.obj.means
                parts = key.split('.')
                joint_id = parts[1]
                param_name = parts[4]
                
                if joint_id != exclude_joint:
                    combined[f"gauss_params.{param_name}"].append(tensor)
                    
                    if param_name == "means":
                        joint_counts[joint_id] = tensor.shape[0]
        
        for joint_id, count in sorted(joint_counts.items()):
            print(f"  [Reconstruct]   {joint_id}: {count:,} Gaussians")
        
        # === 3. Concatenate Everything ===
        result = {}
        for param_key, tensors in combined.items():
            if tensors:
                result[param_key] = torch.cat(tensors, dim=0)
            else:
                # Empty tensor if nothing to concatenate
                param_name = param_key.split('.')[-1]
                if param_name == "means":
                    shape = (0, 3)
                elif param_name == "quats":
                    shape = (0, 4)
                elif param_name == "opacities":
                    shape = (0, 1)
                elif param_name in ["scales", "features_dc"]:
                    shape = (0, 3)
                else:  # features_rest
                    dim_sh = result.get("dim_sh", 1)  # fallback
                    shape = (0, dim_sh - 1, 3)
                
                result[param_key] = torch.empty(shape)
        
        total = result['gauss_params.means'].shape[0]
        print(f"  [Reconstruct] Total reconstructed: {total:,} Gaussians\n")
        
        return result
    
    @staticmethod
    def _convert_legacy_to_new(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert old format to new clean format."""
        new_state = {}
        
        for key, value in state_dict.items():
            if key.startswith("all_gauss_params_obj."):
                # all_gauss_params_obj.joint_0.means -> joints.joint_0.gaussians.obj.means
                parts = key.split('.', 2)
                joint_id = parts[1]
                param_name = parts[2]
                new_state[f"joints.{joint_id}.gaussians.obj.{param_name}"] = value
            
            elif key.startswith("all_gauss_params_canon."):
                parts = key.split('.', 2)
                joint_id = parts[1]
                param_name = parts[2]
                new_state[f"joints.{joint_id}.gaussians.canon.{param_name}"] = value
            
            elif key.startswith("gauss_params_fixed."):
                param_name = key.split('.', 1)[1]
                new_state[f"background.gaussians.{param_name}"] = value
            
            elif key.startswith("all_joint_params."):
                # all_joint_params.joint_0.angles -> joints.joint_0.params.angles
                parts = key.split('.', 2)
                joint_id = parts[1]
                param_name = parts[2]
                
                # Handle old delta format
                if param_name == "angle_deltas":
                    prior_key = f"joint_angles_prior_{joint_id}"
                    if prior_key in state_dict:
                        combined = state_dict[prior_key] + value
                        new_state[f"joints.{joint_id}.params.angles"] = combined
                        print(f"  Converted {joint_id} deltas to angles")
                    else:
                        new_state[f"joints.{joint_id}.params.{param_name}"] = value
                else:
                    new_state[f"joints.{joint_id}.params.{param_name}"] = value
            
            else:
                # Other state (keep as-is)
                new_state[key] = value
        
        return new_state