#!/usr/bin/env python3
"""
Quick Start Example: Run BERT Tracker on a single MOT17 sequence
Shows basic usage and output handling.
"""

import sys
from pathlib import Path
from bert_tracker import BERTTracker

# Add parent to path if needed
sys.path.insert(0, str(Path(__file__).parent))


def example_single_sequence():
    """Example 1: Run tracker on single sequence with minimal setup"""
    
    print("\n" + "="*60)
    print("Example 1: Basic MOT17 Tracking")
    print("="*60)
    
    # Checkpoint auto-detected
    checkpoint = "/home/cs24m118/sam2_backup/checkpoints/sam2_hiera_large.pt"
    
    # Initialize tracker
    tracker = BERTTracker(
        model_cfg="sam2_hiera_l",
        checkpoint=checkpoint,
        device="cuda:0",
        use_drm=True,
    )
    print("✓ Tracker initialized")
    
    # Run full benchmark
    from run_mot17_benchmark import MOT17Benchmark
    
    benchmark = MOT17Benchmark(
        checkpoint=checkpoint,
        device="cuda:0",
    )
    
    # Process single sequence for testing
    metrics = benchmark.run_benchmark(
        split="train",
        sequences=["MOT17-02-FRCNN"]  # Just one sequence for quick test
    )
    
    print("\n" + "="*60)
    print("Benchmark Results")
    print("="*60)
    for seq_name, metric in metrics.items():
        print(f"\n{seq_name}:")
        print(f"  Frames: {metric['tracked_frames']}/{metric['total_frames']}")
        print(f"  Detections: {metric['total_detections']}")
        print(f"  Mask Coverage: {metric.get('average_mask_coverage', 0):.4f}")


def example_programmatic():
    """Example 2: Use BERT tracker programmatically for custom workflow"""
    
    print("\n" + "="*60)
    print("Example 2: Programmatic Usage")
    print("="*60)
    
    import cv2
    import numpy as np
    
    # Initialize
    checkpoint = "/home/cs24m118/sam2_backup/checkpoints/sam2_hiera_large.pt"
    tracker = BERTTracker(
        model_cfg="sam2_hiera_l",
        checkpoint=checkpoint,
        device="cuda:0",
    )
    print("✓ Tracker initialized\n")
    
    # Simulate frame loading
    # frames = [cv2.imread(f"frame_{i:06d}.jpg") for i in range(num_frames)]
    
    # Initialize tracker with first frame and bounding box
    # frame_0 = cv2.imread("frame_000000.jpg")
    # bbox = [100, 100, 200, 150]  # x, y, w, h
    # init_result = tracker.initialize(frame_0, bbox=bbox)
    # obj_id = init_result['obj_id']
    # print(f"✓ Initialized object {obj_id}")
    
    # Track subsequent frames
    # masks_list = []
    # for frame_idx in range(1, len(frames)):
    #     frame = frames[frame_idx]
    #     result = tracker.track(frame)
    #     mask = result['pred_mask']
    #     masks_list.append(mask)
    #     
    #     if (frame_idx + 1) % 10 == 0:
    #         print(f"✓ Tracked frame {frame_idx}")
    
    # Save results
    # For actual implementation, use run_mot17_benchmark.py
    
    print("Example: Load video, initialize, and track frames")
    print("Code skeleton provided above (commented)")


def example_multiple_sequences():
    """Example 3: Process multiple sequences in parallel"""
    
    print("\n" + "="*60)
    print("Example 3: Parallel Processing")
    print("="*60)
    
    from run_bert_tracker_parallel import ParallelBenchmarkRunner
    
    runner = ParallelBenchmarkRunner(
        num_workers=2,  # Use 2 parallel processes
        skip_video=True,  # Skip video generation for speed
    )
    
    # Process 3 sequences in parallel
    results = runner.run(
        split="train",
        sequences=["MOT17-02", "MOT17-04", "MOT17-05"],
        use_pool=True
    )
    
    for seq_name, success, error in results:
        status = "✓" if success else "✗"
        print(f"{status} {seq_name}")


def example_full_benchmark():
    """Example 4: Full MOT17 benchmark"""
    
    print("\n" + "="*60)
    print("Example 4: Full MOT17 Benchmark")
    print("="*60)
    print("\nCommand to run:")
    print("  python run_mot17_benchmark.py --split train --device cuda:0")
    print("\nOptions:")
    print("  --skip-video    : Don't generate videos (faster)")
    print("  --skip-masks    : Don't save masks (faster)")
    print("  --sequences SEQ : Process specific sequences only")
    print("\nExample:")
    print("  python run_mot17_benchmark.py \\")
    print("    --split train \\")
    print("    --sequences MOT17-02-FRCNN MOT17-04-FRCNN \\")
    print("    --device cuda:0")


def main():
    print("\n" + "="*70)
    print(" BERT Tracker - MOT17 Benchmarking Quick Start Guide")
    print("="*70)
    
    print("\n" + "─"*70)
    print("Available Examples:")
    print("─"*70)
    print("1. Single sequence (basic test)")
    print("2. Programmatic usage (custom workflow)")
    print("3. Parallel processing (multiple sequences)")
    print("4. Full benchmark (all sequences)")
    print("5. Run setup verification")
    
    while True:
        choice = input("\nSelect example (1-5) or 'q' to quit: ").strip().lower()
        
        if choice == 'q':
            print("Goodbye!")
            break
        elif choice == '1':
            try:
                example_single_sequence()
            except Exception as e:
                print(f"Error: {e}")
        elif choice == '2':
            example_programmatic()
        elif choice == '3':
            try:
                example_multiple_sequences()
            except Exception as e:
                print(f"Error: {e}")
        elif choice == '4':
            example_full_benchmark()
        elif choice == '5':
            import subprocess
            subprocess.run(["python", "setup_bert_tracker.py"])
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    # If run with argument, skip interactive menu
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "single":
            example_single_sequence()
        elif mode == "programmatic":
            example_programmatic()
        elif mode == "parallel":
            example_multiple_sequences()
        elif mode == "full":
            example_full_benchmark()
        else:
            print(f"Unknown mode: {mode}")
    else:
        main()
