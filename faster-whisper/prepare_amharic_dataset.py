#!/usr/bin/env python3
"""Prepare 150-hour Amharic dataset for Whisper fine-tuning"""

import argparse
import json
from pathlib import Path
from utils.amharic_dataset import AmharicDatasetProcessor
from utils.amharic_tokenizer import AmharicTokenizer

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Amharic dataset for Whisper fine-tuning"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing audio files and transcripts"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to input manifest (JSON or CSV)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/amharic",
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run dataset analysis"
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Filter low-quality samples"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Create train/val/test splits"
    )
    parser.add_argument(
        "--remove_code_switching",
        action="store_true",
        help="Remove samples with code-switching"
    )
    parser.add_argument(
        "--quality_threshold",
        type=float,
        default=0.8,
        help="Minimum quality score (0-1)"
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=1.0,
        help="Minimum audio duration in seconds"
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=25.0,
        help="Maximum audio duration in seconds"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.05,
        help="Test set ratio"
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    print("🚀 Initializing Amharic Dataset Processor...")
    processor = AmharicDatasetProcessor(
        data_dir=args.data_dir,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        quality_threshold=args.quality_threshold
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Analyze dataset
    if args.analyze or not (args.filter or args.split):
        print("\n" + "="*50)
        print("📊 DATASET ANALYSIS")
        print("="*50)
        
        stats = processor.analyze_dataset(args.manifest)
        
        # Print statistics
        print(f"\n📈 Dataset Statistics:")
        print(f"  Total samples: {stats['total_samples']:,}")
        print(f"  Total duration: {stats['total_duration_hours']:.2f} hours")
        print(f"  Average duration: {stats['avg_duration']:.2f}s (σ={stats['std_duration']:.2f}s)")
        print(f"  Median duration: {stats['median_duration']:.2f}s")
        print(f"\n📝 Text Statistics:")
        print(f"  Total characters: {stats['total_characters']:,}")
        print(f"  Total words: {stats['total_words']:,}")
        print(f"  Average text length: {stats['avg_text_length']:.1f} characters")
        print(f"\n🌍 Quality Metrics:")
        print(f"  Code-switching rate: {stats['code_switching_rate']:.1%}")
        print(f"  Quality issues: {stats['quality_issue_rate']:.1%}")
        print(f"  Number of speakers: {stats['num_speakers']}")
        
        print(f"\n⏱️  Duration Distribution:")
        for range_name, count in stats['duration_distribution'].items():
            pct = count / stats['total_samples'] * 100
            print(f"  {range_name}: {count:,} ({pct:.1f}%)")
        
        print(f"\n🗣️  Dialect Distribution:")
        for dialect, count in stats['dialect_distribution'].items():
            pct = count / stats['total_samples'] * 100
            print(f"  {dialect}: {count:,} ({pct:.1f}%)")
        
        # Save statistics
        stats_path = output_dir / "dataset_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Statistics saved to: {stats_path}")
    
    # Step 2: Filter dataset
    manifest_to_split = args.manifest
    
    if args.filter:
        print("\n" + "="*50)
        print("🔧 FILTERING DATASET")
        print("="*50)
        
        filtered_manifest = output_dir / "filtered_manifest.json"
        filter_stats = processor.filter_dataset(
            manifest_path=args.manifest,
            output_path=str(filtered_manifest),
            remove_code_switching=args.remove_code_switching,
            remove_quality_issues=True
        )
        
        print(f"\n📊 Filtering Results:")
        print(f"  Original samples: {filter_stats['original_count']:,}")
        print(f"  Filtered samples: {filter_stats['filtered_count']:,}")
        print(f"  Retention rate: {filter_stats['retention_rate']:.1%}")
        print(f"\n🗑️  Removed:")
        print(f"  Duration issues: {filter_stats['removed_duration']:,}")
        print(f"  Code-switching: {filter_stats['removed_code_switching']:,}")
        print(f"  Quality issues: {filter_stats['removed_quality']:,}")
        print(f"  Other issues: {filter_stats['removed_other']:,}")
        
        manifest_to_split = str(filtered_manifest)
    
    # Step 3: Create splits
    if args.split:
        print("\n" + "="*50)
        print("📂 CREATING TRAIN/VAL/TEST SPLITS")
        print("="*50)
        
        split_stats = processor.create_balanced_splits(
            manifest_path=manifest_to_split,
            output_dir=str(output_dir),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            stratify_by='speaker_id'  # Balance by speaker
        )
        
        print(f"\n✅ Splits created successfully!")
        print(f"  Output directory: {output_dir}")
        print(f"\n📊 Split Statistics:")
        print(f"  Train: {split_stats['train_samples']:,} samples ({split_stats['train_hours']:.1f}h)")
        print(f"  Val:   {split_stats['val_samples']:,} samples ({split_stats['val_hours']:.1f}h)")
        print(f"  Test:  {split_stats['test_samples']:,} samples ({split_stats['test_hours']:.1f}h)")
    
    print("\n" + "="*50)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("="*50)
    print(f"\n📁 Output files in: {output_dir}")
    print("\n🚀 Ready for training! Use the configuration:")
    print("   config/amharic_150h_config.yaml")
    print("\n💡 Next steps:")
    print("   1. Review dataset statistics")
    print("   2. Adjust config/amharic_150h_config.yaml if needed")
    print("   3. Start training: python faster-whisper/train_whisper_lightning.py")

if __name__ == "__main__":
    main()
