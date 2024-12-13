"""
Run all ResNet18 variants in parallel with enhanced logging and progress tracking.

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
import yaml
import signal
from tqdm import tqdm

def format_time(seconds):
    """Format time in seconds to human readable string."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def load_variant_config(config_path):
    """Load variant configuration and return key details."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return {
        'name': Path(config_path).parent.parent.name,
        'description': config.get('variant_description', 'No description'),
        'epochs': config['training']['num_epochs'],
        'batch_size': config['training']['batch_size'],
        'device': config['training']['device']
    }

def run_variant(variant_path, args):
    """Run a single variant training process with enhanced logging.
    
    Args:
        variant_path (str): Path to variant directory
        args: Command line arguments
    """
    config_path = os.path.join(variant_path, 'config', 'hyperparams.yaml')  # Using actual hyperparams
    log_dir = os.path.join(variant_path, 'logs')
    checkpoint_dir = os.path.join(variant_path, 'checkpoints')
    
    # Create necessary directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Get variant details
    variant_info = load_variant_config(config_path)
    
    # Enhanced training command with progress bars and metrics
    train_cmd = [
        'python', '-u',  # Unbuffered output
        'src/train_variant.py',
        config_path,
        '--progress',
        '--log-interval', str(args.log_interval),
        '--save-freq', str(args.save_freq)
    ]
    
    if args.epochs:
        train_cmd.extend(['--epochs', str(args.epochs)])
    
    # Open log file and start process
    with open(log_path, 'w') as log_file:
        # Write header to log file
        log_file.write(f"=== Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_file.write(f"Variant: {variant_info['name']}\n")
        log_file.write(f"Description: {variant_info['description']}\n")
        log_file.write(f"Configuration: {config_path}\n")
        log_file.write(f"Device: {variant_info['device']}\n")
        log_file.write(f"Batch size: {variant_info['batch_size']}\n")
        log_file.write(f"Epochs: {args.epochs or variant_info['epochs']}\n\n")
        log_file.flush()
        
        # Start the training process
        process = subprocess.Popen(
            train_cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            preexec_fn=os.setsid  # Create new process group
        )
        return process, log_path, variant_info

def print_live_logs(variant_name, log_path, lines_shown=0):
    """Print new lines from log file with variant name prefix."""
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            for line in lines[lines_shown:]:
                print(f"{variant_name}: {line.strip()}")
            return len(lines)
    except Exception:
        return lines_shown

def print_status_summary(processes, start_time):
    """Print a summary of all running processes."""
    elapsed = time.time() - start_time
    print(f"\nStatus Update (elapsed time: {format_time(elapsed)})")
    print("=" * 80)
    print(f"{'Variant':20} | {'Status':10} | {'Device':10} | {'Batch Size':10} | {'Description'}")
    print("=" * 80)
    for name, process, _, info in processes:
        status = "Running" if process.poll() is None else "Completed"
        desc = info['description'][:50] + '...' if len(info['description']) > 50 else info['description']
        print(f"{name:20} | {status:10} | {info['device']:10} | {info['batch_size']:<10} | {desc}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Train all ResNet18 variants in parallel')
    parser.add_argument('--variants-dir', type=str, default='experiments/multi_comparison',
                       help='Directory containing variant subdirectories')
    parser.add_argument('--epochs', type=int, help='Override number of epochs for all variants')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='How often to log training metrics (in batches)')
    parser.add_argument('--save-freq', type=int, default=10,
                       help='How often to save checkpoints (in epochs)')
    parser.add_argument('--status-interval', type=int, default=30,
                       help='How often to print status summary (in seconds)')
    args = parser.parse_args()
    
    # Get all variant directories
    variants_dir = Path(args.variants_dir)
    variant_dirs = [d for d in variants_dir.iterdir() if d.is_dir() and d.name.startswith('resnet_variant')]
    variant_dirs.sort()  # Ensure consistent order
    
    print(f"\nFound {len(variant_dirs)} variants to train:")
    for var_dir in variant_dirs:
        config_path = var_dir / 'config' / 'hyperparams.yaml'  # Using actual hyperparams
        info = load_variant_config(config_path)
        print(f"\n- {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Device: {info['device']}")
        print(f"  Batch size: {info['batch_size']}")
        print(f"  Epochs: {args.epochs or info['epochs']}")
    
    # Confirm with user
    input("\nPress Enter to start training all variants...")
    
    # Start all training processes
    processes = []
    log_lines = {}  # Track lines shown for each variant
    start_time = time.time()
    last_status_time = start_time
    
    for var_dir in variant_dirs:
        print(f"\nStarting training for {var_dir.name}")
        process, log_path, info = run_variant(str(var_dir), args)
        processes.append((var_dir.name, process, log_path, info))
        log_lines[var_dir.name] = 0
        time.sleep(2)  # Small delay between launches
    
    # Monitor processes and show live logs
    try:
        while processes:
            current_time = time.time()
            
            # Print status summary periodically
            if current_time - last_status_time >= args.status_interval:
                print_status_summary(processes, start_time)
                last_status_time = current_time
            
            # Update status for all running processes
            for variant_name, process, log_path, _ in processes[:]:
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
                    processes.remove((variant_name, process, log_path, _))
            
            # Brief pause before next update
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nInterrupt received, terminating processes...")
        for _, process, _, _ in processes:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        sys.exit(1)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Print final results summary
    print("\nTraining Complete!")
    print("=" * 60)
    print(f"Total time: {format_time(total_time)}")
    print("\nFinal Results Summary:")
    print("=" * 60)
    
    for var_dir in variant_dirs:
        results_file = var_dir / 'logs' / 'results.txt'
        if results_file.exists():
            with open(results_file, 'r') as f:
                print(f"\n{var_dir.name}:")
                print("-" * 40)
                results = f.readlines()
                # Format results nicely
                for line in results:
                    key, value = line.strip().split(': ')
                    if 'time' in key:
                        value = f"{float(value)/3600:.2f} hours"
                    elif 'accuracy' in key:
                        value = f"{float(value):.2f}%"
                    print(f"{key:20} : {value}")
        else:
            print(f"\n{var_dir.name}: No results file found")
    
    print("\nDetailed logs and reports can be found in each variant's logs directory")

if __name__ == '__main__':
    main() 