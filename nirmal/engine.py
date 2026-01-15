"""
Main data processing engine that orchestrates cleaning, validation, and preprocessing.
"""

import pandas as pd
import numpy as np
import logging
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

from nirmal.cleaners import DataCleaner
from nirmal.validators import DataValidator, ValidationError
from nirmal.preprocessors import DataPreprocessor
from nirmal.config_loader import ConfigLoader
from nirmal.logger_setup import setup_logger


logger = setup_logger()


def _normalize_cleaning_config(cleaning_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize cleaning configuration to handle both boolean and dict formats.
    Converts boolean flags to {enabled: bool} format.
    """
    if not cleaning_config:
        return {}
    
    normalized = cleaning_config.copy()
    
    # List of config keys that might be booleans
    config_keys_to_normalize = [
        'remove_duplicates',
        'handle_missing',
        'handle_invalid_zeros',
        'clean_numeric',
        'outliers',
        'remove_outliers',
        'standardize_text',
        'strip_whitespace'
    ]
    
    for key in config_keys_to_normalize:
        if key in normalized:
            value = normalized[key]
            # If it's a boolean, convert to dict (preserve existing dict if it exists)
            if isinstance(value, bool):
                normalized[key] = {'enabled': value}
            # If it's already a dict but missing 'enabled', add it (preserve other keys)
            elif isinstance(value, dict):
                if 'enabled' not in value:
                    normalized[key] = {'enabled': True, **value}
                # If 'enabled' exists, keep the dict as is
            # If it's None or other type, set to disabled
            elif not isinstance(value, dict):
                normalized[key] = {'enabled': False}
    
    return normalized


def _normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize entire configuration dictionary to handle flexible formats.
    """
    if not config:
        return {}
    
    normalized = config.copy()
    
    # Normalize cleaning config
    if 'cleaning' in normalized:
        normalized['cleaning'] = _normalize_cleaning_config(normalized['cleaning'])
    
    # Normalize pipeline.cleaning if it exists
    if 'pipeline' in normalized and isinstance(normalized['pipeline'], dict):
        if 'cleaning' in normalized['pipeline']:
            # pipeline.cleaning might be a boolean or dict
            cleaning = normalized['pipeline']['cleaning']
            if isinstance(cleaning, bool):
                normalized['pipeline']['cleaning'] = {'enabled': cleaning}
            elif isinstance(cleaning, dict):
                normalized['pipeline']['cleaning'] = _normalize_cleaning_config(cleaning)
    
    return normalized


class DataProcessingEngine:
    """Main engine for orchestrating data processing pipelines."""
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize the data processing engine.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Optional configuration dictionary (overrides config_path)
        """
        self.config: Dict[str, Any] = {}
        
        if config_dict:
            # Normalize config when passed as dict (bypasses ConfigLoader normalization)
            logger.info(f"Normalizing config dict: {config_dict}")
            self.config = _normalize_config(config_dict)
            logger.info(f"Normalized config: {self.config}")
        elif config_path:
            config_loader = ConfigLoader(config_path)
            self.config = config_loader.load()
        else:
            logger.warning("No configuration provided, using empty config")
            self.config = {}
        
        # Initialize processing components
        self.cleaner = DataCleaner(self.config.get('cleaning', {}))
        self.validator = DataValidator(self.config.get('validation', {}))
        self.preprocessor = DataPreprocessor(self.config.get('preprocessing', {}))
        
        # Processing summary tracking
        self.processing_summary = {
            'initial_rows': 0,
            'initial_columns': 0,
            'final_rows': 0,
            'final_columns': 0,
            'missing_values_handled': 0,
            'duplicates_removed': 0,
            'outliers_handled': 0,
            'numeric_columns_validated': 0,
            'text_columns_cleaned': 0,
            'type_coercions': 0
        }
        
        logger.info("DataProcessingEngine initialized")
    
    def process(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Process data through the complete pipeline.
        
        Args:
            input_path: Path to input data file (CSV, Excel, etc.)
            output_path: Optional path to save processed data
        
        Returns:
            Processed DataFrame
        
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValidationError: If validation fails
        """
        logger.info(f"Starting data processing pipeline for: {input_path}")
        
        # Load input data
        df = self._load_data(input_path)
        logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Initialize processing summary
        self.processing_summary = {
            'initial_rows': df.shape[0],
            'initial_columns': df.shape[1],
            'initial_missing': df.isnull().sum().sum(),
            'initial_duplicates': df.duplicated().sum(),
            'final_rows': df.shape[0],
            'final_columns': df.shape[1],
            'missing_values_handled': 0,
            'duplicates_removed': 0,
            'outliers_handled': 0,
            'numeric_columns_validated': 0,
            'text_columns_cleaned': 0,
            'type_coercions': 0
        }
        
        # Get pipeline configuration
        pipeline_config = self.config.get('pipeline', {})
        
        # Step 1: Validation (before cleaning, optional)
        if pipeline_config.get('validate_before_cleaning', False):
            validation_config = self.config.get('validation', {})
            try:
                self.validator.validate(df, validation_config)
            except ValidationError as e:
                logger.error(f"Pre-cleaning validation failed: {e}")
                if pipeline_config.get('strict_validation', False):
                    raise
        
        # Step 2: Data Cleaning
        cleaning_enabled = pipeline_config.get('cleaning', {})
        if isinstance(cleaning_enabled, dict):
            cleaning_enabled = cleaning_enabled.get('enabled', True)
        elif isinstance(cleaning_enabled, bool):
            pass  # Already a boolean
        else:
            cleaning_enabled = True  # Default to enabled
        
        if cleaning_enabled:
            cleaning_config = self.config.get('cleaning', {})
            rows_before = len(df)
            missing_before = df.isnull().sum().sum()
            
            # Log cleaning config for debugging
            logger.info(f"Cleaning enabled: {cleaning_enabled}")
            logger.info(f"Cleaning config keys: {list(cleaning_config.keys())}")
            logger.info(f"handle_missing config: {cleaning_config.get('handle_missing', 'NOT FOUND')}")
            logger.info(f"remove_duplicates config: {cleaning_config.get('remove_duplicates', 'NOT FOUND')}")
            
            df = self.cleaner.clean(df, cleaning_config)
            
            # Update processing summary
            self.processing_summary['duplicates_removed'] = rows_before - len(df)
            self.processing_summary['missing_values_handled'] = missing_before - df.isnull().sum().sum()
            self.processing_summary['final_rows'] = len(df)
            self.processing_summary['final_columns'] = len(df.columns)
            
            # Count numeric and text columns processed
            numeric_cols = df.select_dtypes(include=['number']).columns
            text_cols = df.select_dtypes(include=['object']).columns
            self.processing_summary['numeric_columns_validated'] = len(numeric_cols)
            self.processing_summary['text_columns_cleaned'] = len(text_cols)
            
            logger.info(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Step 3: Validation (after cleaning)
        if pipeline_config.get('validation', {}).get('enabled', True):
            validation_config = self.config.get('validation', {})
            try:
                self.validator.validate(df, validation_config)
                logger.info("Post-cleaning validation passed")
            except ValidationError as e:
                logger.error(f"Post-cleaning validation failed: {e}")
                if pipeline_config.get('strict_validation', False):
                    raise
        
        # Step 4: Preprocessing
        if pipeline_config.get('preprocessing', {}).get('enabled', True):
            preprocessing_config = self.config.get('preprocessing', {})
            df = self.preprocessor.preprocess(df, preprocessing_config)
            logger.info(f"After preprocessing: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Step 5: Save output
        if output_path:
            self._save_data(df, output_path)
            logger.info(f"Processed data saved to: {output_path}")
        
        # Finalize processing summary
        self.processing_summary['final_rows'] = df.shape[0]
        self.processing_summary['final_columns'] = df.shape[1]
        
        logger.info("Data processing pipeline completed successfully")
        logger.info(f"Processing Summary: {self.processing_summary['missing_values_handled']} missing values handled, "
                   f"{self.processing_summary['duplicates_removed']} duplicates removed, "
                   f"{self.processing_summary['numeric_columns_validated']} numeric columns validated, "
                   f"{self.processing_summary['text_columns_cleaned']} text columns cleaned")
        return df
    
    def _load_data(self, input_path: str) -> pd.DataFrame:
        """
        Load data from various file formats.
        
        Args:
            input_path: Path to input file
        
        Returns:
            Loaded DataFrame
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        path = Path(input_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        file_extension = path.suffix.lower()
        
        try:
            if file_extension == '.csv':
                # Read CSV and convert empty strings to NaN
                df = pd.read_csv(input_path, na_values=['', ' ', 'nan', 'NaN', 'NULL', 'null'])
                # Also handle whitespace-only strings
                df = df.replace(r'^\s*$', np.nan, regex=True)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(input_path)
            elif file_extension == '.json':
                df = pd.read_json(input_path)
            elif file_extension == '.parquet':
                df = pd.read_parquet(input_path)
            elif file_extension in ['.yaml', '.yml']:
                # Load YAML file as data
                with open(input_path, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f)
                
                # Convert YAML to DataFrame
                # Use dtype='object' initially to prevent aggressive type inference
                # This avoids ArrowInvalid errors when columns have mixed types
                if isinstance(yaml_data, list):
                    # If YAML is a list of dictionaries, convert directly
                    if len(yaml_data) > 0 and isinstance(yaml_data[0], dict):
                        df = pd.DataFrame(yaml_data, dtype='object')
                    else:
                        df = pd.DataFrame(yaml_data, dtype='object')
                elif isinstance(yaml_data, dict):
                    # If YAML is a dictionary, try to find a data key or convert directly
                    if 'data' in yaml_data and isinstance(yaml_data['data'], list):
                        df = pd.DataFrame(yaml_data['data'], dtype='object')
                    elif 'metadata' in yaml_data and 'data' in yaml_data:
                        # Handle our saved YAML format with metadata
                        df = pd.DataFrame(yaml_data['data'], dtype='object')
                    elif 'records' in yaml_data and isinstance(yaml_data['records'], list):
                        # Handle records format
                        df = pd.DataFrame(yaml_data['records'], dtype='object')
                    else:
                        # Try to flatten the dict structure
                        # If dict contains lists of dicts, use those
                        list_values = [v for v in yaml_data.values() if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict)]
                        if list_values:
                            df = pd.DataFrame(list_values[0], dtype='object')
                        else:
                            # Convert simple dict to DataFrame (each key becomes a column)
                            # Handle nested dicts by flattening using json_normalize
                            try:
                                df = pd.json_normalize(yaml_data)
                                # Ensure object dtype to prevent type coercion issues
                                df = df.astype('object')
                            except Exception as e:
                                logger.warning(f"Could not normalize YAML dict structure: {e}. Attempting simple conversion.")
                                # Fallback: convert to single row, handling nested structures
                                try:
                                    df = pd.DataFrame([yaml_data], dtype='object')
                                except Exception as e2:
                                    raise ValueError(f"Cannot convert YAML dictionary to DataFrame. Structure may be too complex. Error: {e2}")
                else:
                    raise ValueError(f"YAML file must contain a list or dictionary, got {type(yaml_data)}")
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            logger.info(f"Successfully loaded {file_extension} file")
            return df
        
        except Exception as e:
            logger.error(f"Error loading data file: {e}")
            raise
    
    def _save_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save DataFrame to various file formats.
        
        Args:
            df: DataFrame to save
            output_path: Path to output file
        
        Raises:
            ValueError: If file format is not supported
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        file_extension = path.suffix.lower()
        
        try:
            if file_extension == '.csv':
                # Save CSV - NaN values will be written as empty strings (standard CSV behavior)
                df.to_csv(output_path, index=False, na_rep='')
            elif file_extension in ['.xlsx', '.xls']:
                df.to_excel(output_path, index=False)
            elif file_extension == '.json':
                df.to_json(output_path, orient='records', indent=2)
            elif file_extension == '.parquet':
                df.to_parquet(output_path, index=False)
            elif file_extension in ['.yaml', '.yml']:
                # Convert DataFrame to YAML format
                # Convert DataFrame to list of dictionaries
                data_list = df.to_dict('records')
                
                # Create YAML structure with metadata
                yaml_data = {
                    'metadata': {
                        'rows': len(df),
                        'columns': len(df.columns),
                        'column_names': list(df.columns)
                    },
                    'data': data_list
                }
                
                # Write YAML file
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            logger.info(f"Successfully saved {file_extension} file")
        
        except Exception as e:
            logger.error(f"Error saving data file: {e}")
            raise
    
    def get_validation_results(self) -> list:
        """Get validation results from the last run."""
        return self.validator.get_results()
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get processing summary statistics from the last run."""
        return self.processing_summary.copy()