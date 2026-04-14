#!/usr/bin/env python3
"""
BERT Tracker Parallel Benchmark Runner
Runs tracking on multiple sequences in parallel to speed up processing.
"""

import subprocess
import multiprocessing
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParallelBenchmarkRunner:
    """Run benchmark on multiple sequences in parallel"""
    
    def __init__(
        self,
        dataset_root="/home/cs24m118/datasets/MOT17",
        output_root="/home/cs24m118/phase2/mysam2/mot17_results",
        num_workers=2,
        skip_video=False,
        skip_masks=False,
    ):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.num_workers = num_workers
        self.skip_video = skip_video
        self.skip_masks = skip_masks

    def get_sequences(self, split="train"):
        """Get all sequences"""
        split_dir = self.dataset_root / split
        sequences = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        return sequences

    def run_sequence(self, sequence_name, worker_id):
        """Run tracking on single sequence"""
        logger.info(f"[Worker {worker_id}] Processing {sequence_name}...")
        
        cmd = [
            "python", "run_mot17_benchmark.py",
            "--split", "train",
            "--sequences", sequence_name,
        ]
        
        if self.skip_video:
            cmd.append("--skip-video")
        
        if self.skip_masks:
            cmd.append("--skip-masks")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode == 0:
                logger.info(f"[Worker {worker_id}] ✓ Completed {sequence_name}")
                return (sequence_name, True, None)
            else:
                logger.error(f"[Worker {worker_id}] ✗ Failed {sequence_name}")
                return (sequence_name, False, result.stderr)
        except subprocess.TimeoutExpired:
            logger.error(f"[Worker {worker_id}] Timeout on {sequence_name}")
            return (sequence_name, False, "Timeout")
        except Exception as e:
            logger.error(f"[Worker {worker_id}] Error on {sequence_name}: {e}")
            return (sequence_name, False, str(e))

    def process_pool(self, sequences):
        """Process sequences in parallel using pool"""
        logger.info(f"Processing {len(sequences)} sequences with {self.num_workers} workers")
        
        results = []
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            async_results = []
            
            # Submit all tasks
            for worker_id, seq in enumerate(sequences):
                async_result = pool.apply_async(
                    self.run_sequence,
                    args=(seq, worker_id % self.num_workers)
                )
                async_results.append(async_result)
            
            # Collect results
            for async_result in async_results:
                try:
                    result = async_result.get(timeout=3600)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error collecting result: {e}")
        
        return results

    def run(self, split="train", sequences=None, use_pool=True):
        """Run benchmark"""
        logger.info("="*60)
        logger.info("Parallel BERT Tracker Benchmark")
        logger.info("="*60)
        
        all_sequences = self.get_sequences(split)
        
        if sequences:
            all_sequences = [s for s in all_sequences if any(seq in s for seq in sequences)]
        
        logger.info(f"Processing {len(all_sequences)} sequences on {split} split")
        
        if use_pool and self.num_workers > 1:
            results = self.process_pool(all_sequences)
        else:
            results = [self.run_sequence(seq, 0) for seq in all_sequences]
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("Benchmark Summary")
        logger.info("="*60)
        
        successful = sum(1 for _, success, _ in results if success)
        failed = sum(1 for _, success, _ in results if not success)
        
        logger.info(f"Total: {len(results)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        
        if failed > 0:
            logger.warning("\nFailed sequences:")
            for seq, success, error in results:
                if not success:
                    logger.warning(f"  - {seq}: {error[:100] if error else 'Unknown error'}")
        
        logger.info("\n" + "="*60)
        return results


def main():
    parser = argparse.ArgumentParser(description="Parallel BERT Tracker Benchmark")
    parser.add_argument('--split', default="train", help="Dataset split (train/test)")
    parser.add_argument('--sequences', nargs='+', default=None, help="Specific sequences")
    parser.add_argument('--workers', type=int, default=2, help="Number of parallel workers")
    parser.add_argument('--skip-video', action='store_true', help="Skip video generation")
    parser.add_argument('--skip-masks', action='store_true', help="Skip mask saving")
    parser.add_argument('--output-root', default="/home/cs24m118/phase2/mysam2/mot17_results")
    parser.add_argument('--dataset-root', default="/home/cs24m118/datasets/MOT17")
    
    args = parser.parse_args()
    
    runner = ParallelBenchmarkRunner(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        num_workers=args.workers,
        skip_video=args.skip_video,
        skip_masks=args.skip_masks,
    )
    
    results = runner.run(
        split=args.split,
        sequences=args.sequences,
        use_pool=(args.workers > 1)
    )
    
    return results


if __name__ == "__main__":
    main()
