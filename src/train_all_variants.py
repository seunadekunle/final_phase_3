"""
Run all ResNet18 variants in parallel.

This script launches separate processes for each variant, allowing them to train
concurrently while managing their outputs independently.
"""

import os
import sys
import subprocess
from pathlib import Path
import time
import argparse
from datetime import datetime

def format_time(seconds):
    """Format time in seconds to human readable string."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def run_variant(variant_path):
    """Run a single variant training process with enhanced logging.
    
    Args:
        variant_path (str): Path to variant directory
    """
    config_path = os.path.join(variant_path, 'config', 'test_hyperparams.yaml')
    log_dir = os.path.join(variant_path, 'logs')
    log_path = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Enhanced training command with progress bars and metrics
    train_cmd = [
        'python', '-u',  # Unbuffered output
        'src/train_variant.py',
        config_path,
        '--progress',  # Enable progress bars
        '--log-interval', '1',  # Log every batch
        '--metrics', 'loss,acc,lr',  # Track these metrics
        '--save-freq', '1'  # Save checkpoints every epoch
    ]
    
    # Open log file and start process
    with open(log_path, 'w') as log_file:
        # Write header to log file
        log_file.write(f"=== Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_file.write(f"Configuration: {config_path}\n\n")
        log_file.flush()
        
        # Start the training process
        process = subprocess.Popen(
            train_cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )
        return process, log_path

def print_live_logs(variant_name, log_path, lines_shown=0):
    """Print new lines from log file."""
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in lines[lines_shown:]:
                print(f"{variant_name}: {line.strip()}")
            return len(lines)
    except Exception:
        return lines_shown

def main():
    parser = argparse.ArgumentParser(description='Train all ResNet18 variants in parallel')
    parser.add_argument('--variants-dir', type=str, default='experiments/multi_comparison',
                       help='Directory containing variant subdirectories')
    args = parser.parse_args()
    
    # Get all variant directories
    variants_dir = Path(args.variants_dir)
    variant_dirs = [d for d in variants_dir.iterdir() if d.is_dir() and d.name.startswith('resnet_variant')]
    variant_dirs.sort()  # Ensure consistent order
    
    print(f"\nFound {len(variant_dirs)} variants to train:")
    for var_dir in variant_dirs:
        print(f"- {var_dir.name}")
    
    # Start all training processes
    processes = []
    log_lines = {}  # Track lines shown for each variant
    start_time = time.time()
    
    for var_dir in variant_dirs:
        print(f"\nStarting training for {var_dir.name}")
        process, log_path = run_variant(str(var_dir))
        processes.append((var_dir.name, process, log_path))
        log_lines[var_dir.name] = 0
        time.sleep(2)  # Small delay between launches
    
    # Monitor processes and show live logs
    try:
        while processes:
            # Update status for all running processes
            for variant_name, process, log_path in processes[:]:
                # Check process status
                return_code = process.poll()
                
                # Update log output
                log_lines[variant_name] = print_live_logs(
                    variant_name, log_path, log_lines[variant_name]
                )
                
                if return_code is not None:
                    if return_code == 0:
                        print(f"\n{variant_name} training completed successfully")
                    else:
                        print(f"\n{variant_name} training failed with return code {return_code}")
                    processes.remove((variant_name, process, log_path))
            
            # Brief pause before next update
            time.sleep(5)
            
            # Print summary status
            if processes:
                elapsed = time.time() - start_time
                print(f"\nStatus update (elapsed time: {format_time(elapsed)}):")
                for variant_name, _, _ in processes:
                    print(f"- {variant_name}: Running")
    
    except KeyboardInterrupt:
        print("\nInterrupt received, terminating processes...")
        for _, process, _ in processes:
            process.terminate()
        sys.exit(1)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Print final results summary
    print("\nTraining Complete!")
    print("="*60)
    print(f"Total time: {format_time(total_time)}")
    print("\nFinal Results Summary:")
    print("="*60)
    
    for var_dir in variant_dirs:
        results_file = var_dir / 'logs' / 'results.txt'
        if results_file.exists():
            with open(results_file, 'r') as f:
                print(f"\n{var_dir.name}:")
                print("-" * 40)
                print(f.read())
        else:
            print(f"\n{var_dir.name}: No results file found")
    
    print("\nDetailed logs and reports can be found in each variant's logs directory")

if __name__ == '__main__':
    main() 