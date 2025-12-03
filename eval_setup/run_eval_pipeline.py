#!/usr/bin/env python3
import yaml
import subprocess
import json
import time
from pathlib import Path
import argparse
import tempfile


def run_pipeline_batch(eval_config_path, eval_data_root):
    """Run TwinSplat pipeline on all evaluation objects"""
    
    with open(eval_config_path) as f:
        config = yaml.safe_load(f)

    change_cfg = config.get("change_detection", {})
    tmp_cfg_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(change_cfg, tmp_cfg_file, indent=2)
    tmp_cfg_file.close()
    change_cfg_path = tmp_cfg_file.name

    results = []
    eval_data_root = Path(eval_data_root)
    project_root = Path(__file__).resolve().parent.parent
    script_path = project_root / "arti_pipeline_sim.sh"
        
    for obj in config['datasets']['partnet_objects']:
        object_id = obj['object_id']
        category = obj['category']
        joint_id = obj['joint_id']
        
        print(f"\n{'='*60}")
        print(f"Processing: {category}/{object_id}/{joint_id}")
        print(f"{'='*60}")
        
        # Check if data exists
        data_dir = eval_data_root / f"{category}_{object_id}"
        if not data_dir.exists():
            # fallback to flat structure
            alt_dir = eval_data_root / object_id
            if alt_dir.exists():
                data_dir = alt_dir
            else:
                print(f"❌ Data not found: {data_dir} or {alt_dir}")
                results.append({
                    'object_id': object_id,
                    'category': category,
                    'joint_id': joint_id,
                    'status': 'data_missing'
                })
                continue
        
        start_time = time.time()
        
        try:
            # Run pipeline - skip rendering since data already exists
            subprocess.run(
                [
                    str(script_path),
                    str(data_dir),
                    '0',
                    'true',
                    'false',
                    change_cfg_path,
                ],
                check=True,
                cwd=str(project_root)    # ← IMPORTANT
            )

            end_time = time.time()
            runtime = end_time - start_time
            
            print(f"✅ Success: {object_id} in {runtime:.1f}s")
            results.append({
                'object_id': object_id,
                'category': category,
                'joint_id': joint_id,
                'status': 'success',
                'runtime_seconds': runtime,
                'output_dir': f'outputs_sim/{object_id}'
            })
            
        except subprocess.CalledProcessError as e:
            end_time = time.time()
            print(f"❌ Failed: {object_id} - {e}")
            results.append({
                'object_id': object_id,
                'category': category, 
                'joint_id': joint_id,
                'status': 'failed',
                'error': str(e),
                'runtime_seconds': end_time - start_time
            })
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f'eval_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BATCH COMPLETE")
    print(f"Results saved: {results_file}")
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"Success rate: {success_count}/{len(results)}")
    print(f"{'='*60}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_config', default='configs/eval_config.yaml')
    parser.add_argument('--eval_data_root', required=True) 
    args = parser.parse_args()
    
    run_pipeline_batch(args.eval_config, args.eval_data_root)