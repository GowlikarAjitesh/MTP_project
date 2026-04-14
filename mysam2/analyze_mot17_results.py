#!/usr/bin/env python3
"""
MOT17 Detailed Benchmark Analysis
Analyzes and reports detailed metrics from tracker outputs
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MOT17AnalysisReport:
    """Generate detailed benchmark analysis reports"""
    
    def __init__(self, results_dir="/home/cs24m118/phase2/mysam2/mot17_results"):
        self.results_dir = Path(results_dir)
        self.metrics_dir = self.results_dir / "metrics"
        self.tracks_dir = self.results_dir / "tracks"
    
    def load_summary(self):
        """Load summary metrics"""
        summary_file = self.metrics_dir / "summary.json"
        if not summary_file.exists():
            logger.error(f"Summary file not found: {summary_file}")
            return None
        
        with open(summary_file) as f:
            return json.load(f)
    
    def generate_report(self):
        """Generate comprehensive benchmarking report"""
        summary = self.load_summary()
        if not summary:
            return
        
        print("\n" + "="*70)
        print(" MOT17 BERT TRACKER - DETAILED BENCHMARKING REPORT")
        print("="*70)
        
        print(f"\nTimestamp: {summary['timestamp']}")
        print(f"Total Sequences: {summary['total_sequences']}")
        print(f"Successful: {summary['successful_sequences']}")
        print(f"Failed: {summary['failed_sequences']}")
        
        if summary['failed_sequences'] > 0:
            print(f"\nFailed Sequences:")
            for seq_name in summary['failed_names']:
                print(f"  - {seq_name}")
        
        # Aggregate metrics
        if 'aggregate' in summary:
            print("\n" + "-"*70)
            print("AGGREGATE PERFORMANCE METRICS")
            print("-"*70)
            agg = summary['aggregate']
            
            print(f"\nFrame Statistics:")
            print(f"  Total Frames:      {agg['total_frames']:,}")
            print(f"  Tracked Frames:    {agg['tracked_frames']:,}")
            print(f"  Tracking Success:  {agg['tracking_ratio']*100:.1f}%")
            
            print(f"\nDetection Statistics:")
            print(f"  Total Detections:         {agg['total_detections']:,}")
            print(f"  Avg/Frame:                {agg['average_detections_per_frame']:.2f}")
            print(f"  Avg Detection Conf:       {agg['average_detection_confidence']:.4f}")
            
            print(f"\nSegmentation Statistics:")
            print(f"  Avg Mask Coverage:        {agg['average_mask_coverage']:.4f} ({agg['average_mask_coverage']*100:.2f}%)")
            print(f"  Total Mask Pixels:        {agg['total_mask_pixels']:,}")
        
        # Per-sequence details
        print("\n" + "-"*70)
        print("PER-SEQUENCE BREAKDOWN")
        print("-"*70)
        print(f"\n{'Sequence':<25} {'Frames':<10} {'Detect':<10} {'Conf':<8} {'Coverage':<10}")
        print("-"*70)
        
        for seq_name, metrics in sorted(summary['sequences'].items()):
            frames = f"{metrics['tracked_frames']}/{metrics['total_frames']}"
            dets = f"{metrics['total_detections']}"
            conf = f"{metrics.get('average_confidence', 0):.4f}"
            coverage = f"{metrics.get('average_mask_coverage', 0):.4f}"
            
            print(f"{seq_name:<25} {frames:<10} {dets:<10} {conf:<8} {coverage:<10}")
        
        print("\n" + "="*70)
        print("REPORT GENERATION COMPLETE")
        print("="*70 + "\n")
    
    def analyze_by_detector_type(self):
        """Analyze performance by detector family (DPM, FRCNN, SDP)"""
        summary = self.load_summary()
        if not summary:
            return
        
        detector_stats = defaultdict(lambda: {
            'sequences': [],
            'frames': 0,
            'detections': 0,
            'confidences': [],
            'coverages': []
        })
        
        for seq_name, metrics in summary['sequences'].items():
            # Extract detector type (e.g., MOT17-02-FRCNN -> FRCNN)
            parts = seq_name.split('-')
            if len(parts) >= 3:
                detector_type = parts[2]
                stats = detector_stats[detector_type]
                stats['sequences'].append(seq_name)
                stats['frames'] += metrics['total_frames']
                stats['detections'] += metrics['total_detections']
                stats['confidences'].append(metrics.get('average_confidence', 0))
                stats['coverages'].append(metrics.get('average_mask_coverage', 0))
        
        print("\n" + "="*70)
        print("PERFORMANCE BY DETECTOR TYPE")
        print("="*70)
        
        for detector_type in ['DPM', 'FRCNN', 'SDP']:
            if detector_type in detector_stats:
                stats = detector_stats[detector_type]
                print(f"\n{detector_type} Detector:")
                print(f"  Sequences:           {len(stats['sequences'])}")
                print(f"  Total Frames:        {stats['frames']:,}")
                print(f"  Total Detections:    {stats['detections']:,}")
                print(f"  Avg Confidence:      {np.mean(stats['confidences']):.4f}")
                print(f"  Avg Coverage:        {np.mean(stats['coverages']):.4f}")
                print(f"  Sequences: {', '.join(stats['sequences'][:3])}")
                if len(stats['sequences']) > 3:
                    print(f"            ... + {len(stats['sequences'])-3} more")
    
    def print_detailed_metrics(self):
        """Print detailed metrics for each sequence"""
        summary = self.load_summary()
        if not summary:
            return
        
        print("\n" + "="*70)
        print("DETAILED PER-SEQUENCE METRICS")
        print("="*70)
        
        for seq_name in sorted(summary['sequences'].keys()):
            metrics = summary['sequences'][seq_name]
            print(f"\n{seq_name}:")
            print(f"  Frames:                 {metrics['tracked_frames']}/{metrics['total_frames']}")
            print(f"  Total Detections:       {metrics['total_detections']:,}")
            print(f"  Frames with Detections: {metrics.get('frames_with_detections', 'N/A')}")
            print(f"  Avg Detections/Frame:   {metrics.get('average_detections_per_frame', 0):.2f}")
            print(f"  Max Detections/Frame:   {metrics.get('max_detections_in_frame', 0)}")
            print(f"  Confidence Min/Avg/Max: {metrics.get('min_confidence', 0):.4f} / {metrics.get('average_confidence', 0):.4f} / {metrics.get('max_confidence', 0):.4f}")
            print(f"  Mask Coverage:          {metrics.get('average_mask_coverage', 0):.4f}")
            print(f"  Total Mask Pixels:      {metrics.get('total_mask_pixels', 0):,}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="MOT17 Benchmark Analysis")
    parser.add_argument('--results-dir', default="/home/cs24m118/phase2/mysam2/mot17_results",
                       help="Results directory")
    parser.add_argument('--detailed', action='store_true', help="Show detailed metrics")
    parser.add_argument('--by-detector', action='store_true', help="Analyze by detector type")
    
    args = parser.parse_args()
    
    analyzer = MOT17AnalysisReport(args.results_dir)
    
    # Always print main report
    analyzer.generate_report()
    
    if args.by_detector:
        analyzer.analyze_by_detector_type()
    
    if args.detailed:
        analyzer.print_detailed_metrics()


if __name__ == "__main__":
    main()
