"""
Main entry point for the athlete dataset processing pipeline.
"""

import os
import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from scripts.process_sport import process_sport_data
from scripts.merge_sports import merge_all_sports, create_sport_comparison_report


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Athlete Dataset Processing Pipeline")
    parser.add_argument("--sport", type=str, help="Process specific sport only")
    parser.add_argument("--merge-only", action="store_true", help="Only run merge step")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merge step")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    
    args = parser.parse_args()
    
    # Define paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    processed_dir = base_dir / "processed"
    output_dir = Path(args.output_dir)
    
    # Create output directories
    processed_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Define sports to process
    sports = [
        "basketball", "swimming", "soccer", "tennis", "track_field",
        "gymnastics", "volleyball", "baseball", "hockey", "cycling"
    ]
    
    print("=" * 60)
    print("ATHLETE DATASET PROCESSING PIPELINE")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Process individual sports (unless merge-only)
    if not args.merge_only:
        print("STEP 1: Processing individual sports")
        print("-" * 40)
        
        sports_to_process = [args.sport] if args.sport else sports
        processing_results = {}
        
        for sport in sports_to_process:
            raw_file = data_dir / f"{sport}_raw.csv"
            processed_file = processed_dir / f"{sport}.csv"
            
            if not raw_file.exists():
                print(f"⚠️  Raw data file not found: {raw_file}")
                continue
            
            result = process_sport_data(
                sport_name=sport,
                raw_data_path=str(raw_file),
                output_path=str(processed_file)
            )
            
            processing_results[sport] = result
            
            if "error" in result:
                print(f"❌ {sport}: {result['error']}")
            else:
                print(f"✅ {sport}: {result['original_records']} → {result['processed_records']} records")
                print(f"   Quality score: {result['validation']['quality_score']:.1f}%")
        
        print()
        
        # Print processing summary
        successful = [s for s, r in processing_results.items() if "error" not in r]
        failed = [s for s, r in processing_results.items() if "error" in r]
        
        print(f"Processing complete: {len(successful)} successful, {len(failed)} failed")
        if failed:
            print(f"Failed sports: {', '.join(failed)}")
        print()
    
    # Merge all sports (unless skip-merge)
    if not args.skip_merge:
        print("STEP 2: Merging all sports data")
        print("-" * 40)
        
        merged_file = output_dir / "athlete_dataset_merged.csv"
        metadata_file = output_dir / "dataset_metadata.json"
        comparison_file = output_dir / "sport_comparison.json"
        
        merge_result = merge_all_sports(
            processed_dir=str(processed_dir),
            output_path=str(merged_file),
            metadata_path=str(metadata_file)
        )
        
        if "error" in merge_result:
            print(f"❌ Merge failed: {merge_result['error']}")
            return 1
        
        print(f"✅ Merged dataset created: {merge_result['total_records']} total records")
        print(f"   Sports included: {', '.join(merge_result['sports_included'])}")
        print(f"   Output file: {merged_file}")
        print(f"   Metadata file: {metadata_file}")
        
        # Create sport comparison report
        try:
            import pandas as pd
            merged_df = pd.read_csv(merged_file)
            comparison = create_sport_comparison_report(merged_df, str(comparison_file))
            print(f"   Comparison report: {comparison_file}")
        except Exception as e:
            print(f"⚠️  Could not create comparison report: {str(e)}")
        
        print()
    
    print("=" * 60)
    print(f"Pipeline complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    return 0


def process_single_sport(sport_name: str, data_dir: str = None, output_dir: str = None):
    """
    Convenience function to process a single sport.
    
    Args:
        sport_name: Name of the sport to process
        data_dir: Directory containing raw data (default: ./data)
        output_dir: Directory for processed output (default: ./processed)
    """
    base_dir = Path(__file__).parent
    data_dir = Path(data_dir) if data_dir else base_dir / "data"
    output_dir = Path(output_dir) if output_dir else base_dir / "processed"
    
    raw_file = data_dir / f"{sport_name}_raw.csv"
    processed_file = output_dir / f"{sport_name}.csv"
    
    if not raw_file.exists():
        print(f"Error: Raw data file not found: {raw_file}")
        return None
    
    output_dir.mkdir(exist_ok=True)
    
    result = process_sport_data(
        sport_name=sport_name,
        raw_data_path=str(raw_file),
        output_path=str(processed_file)
    )
    
    return result


def quick_merge(processed_dir: str = None, output_file: str = None):
    """
    Convenience function to quickly merge all processed sports.
    
    Args:
        processed_dir: Directory containing processed CSV files (default: ./processed)
        output_file: Output file path (default: ./athlete_dataset_merged.csv)
    """
    base_dir = Path(__file__).parent
    processed_dir = processed_dir or str(base_dir / "processed")
    output_file = output_file or str(base_dir / "athlete_dataset_merged.csv")
    
    result = merge_all_sports(
        processed_dir=processed_dir,
        output_path=output_file
    )
    
    return result


if __name__ == "__main__":
    sys.exit(main())
