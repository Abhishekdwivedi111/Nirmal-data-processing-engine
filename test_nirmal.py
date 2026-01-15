"""
Test script for Nirmal Data Processing Engine.
This script demonstrates how to use the engine programmatically.
"""

import pandas as pd
from pathlib import Path
from nirmal.engine import DataProcessingEngine
from nirmal.logger_setup import setup_logger

# Setup logger
logger = setup_logger(log_level="INFO")


def test_basic_processing():
    """Test basic data processing pipeline."""
    print("=" * 70)
    print("Testing Nirmal Data Processing Engine")
    print("=" * 70)
    
    # Check if sample data exists
    sample_data_path = "data/sample_input.csv"
    if not Path(sample_data_path).exists():
        print(f"\n❌ Sample data not found: {sample_data_path}")
        print("Please run 'python create_sample_data.py' first to create sample data.")
        return False
    
    # Load and display original data
    print("\n📊 Original Data:")
    print("-" * 70)
    df_original = pd.read_csv(sample_data_path)
    print(f"Shape: {df_original.shape[0]} rows, {df_original.shape[1]} columns")
    print(f"Missing values:\n{df_original.isnull().sum()}")
    print(f"Duplicate rows: {df_original.duplicated().sum()}")
    print(f"\nFirst 5 rows:")
    print(df_original.head())
    
    try:
        # Initialize engine
        print("\n🔧 Initializing Data Processing Engine...")
        config_path = "config/test_config.yaml"
        
        if not Path(config_path).exists():
            print(f"\n❌ Configuration file not found: {config_path}")
            print("Using example_config.yaml instead...")
            config_path = "config/example_config.yaml"
        
        engine = DataProcessingEngine(config_path=config_path)
        
        # Process data
        print("\n⚙️  Processing data...")
        output_path = "data/sample_output.csv"
        df_processed = engine.process(sample_data_path, output_path)
        
        # Display results
        print("\n✅ Processing Complete!")
        print("-" * 70)
        print(f"Processed Data Shape: {df_processed.shape[0]} rows, {df_processed.shape[1]} columns")
        print(f"Missing values:\n{df_processed.isnull().sum()}")
        print(f"Duplicate rows: {df_processed.duplicated().sum()}")
        print(f"\nFirst 5 rows of processed data:")
        print(df_processed.head())
        
        # Display validation results
        validation_results = engine.get_validation_results()
        if validation_results:
            print("\n📋 Validation Results:")
            print("-" * 70)
            for result in validation_results:
                status = "✅ PASSED" if result['status'] == 'passed' else "❌ FAILED"
                print(f"  {status}: {result['check']}")
                if 'errors' in result and result['errors']:
                    for error in result['errors'][:3]:  # Show first 3 errors
                        print(f"    - {error}")
        
        print(f"\n💾 Output saved to: {output_path}")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        logger.exception("Full error details:")
        return False


if __name__ == "__main__":
    success = test_basic_processing()
    exit(0 if success else 1)
