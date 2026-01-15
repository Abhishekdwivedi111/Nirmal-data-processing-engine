"""
Main entry point for Nirmal Data Processing Engine.
"""

import argparse
import sys
from pathlib import Path

from nirmal.engine import DataProcessingEngine
from nirmal.logger_setup import setup_logger


def main():
    """Main function to run the data processing engine."""
    parser = argparse.ArgumentParser(
        description='Nirmal - Intelligent Data Processing Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config config/example_config.yaml --input data/input.csv --output data/output.csv
  python main.py --config config/example_config.yaml --input data/input.xlsx --output data/output.csv
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input data file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to output data file (optional)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger(log_level=args.log_level)
    
    try:
        # Initialize engine
        logger.info("Initializing Nirmal Data Processing Engine")
        engine = DataProcessingEngine(config_path=args.config)
        
        # Process data
        df = engine.process(args.input, args.output)
        
        # Display summary
        logger.info("=" * 60)
        logger.info("Processing Summary")
        logger.info("=" * 60)
        logger.info(f"Input file: {args.input}")
        if args.output:
            logger.info(f"Output file: {args.output}")
        logger.info(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info("=" * 60)
        
        # Display validation results
        validation_results = engine.get_validation_results()
        if validation_results:
            logger.info("Validation Results:")
            for result in validation_results:
                status_symbol = "✓" if result['status'] == 'passed' else "✗"
                logger.info(f"  {status_symbol} {result['check']}: {result['status']}")
        
        logger.info("Processing completed successfully!")
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
