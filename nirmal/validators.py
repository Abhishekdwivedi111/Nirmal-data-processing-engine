"""
Data validation modules for data quality checks.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Callable


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class DataValidator:
    """Data validation operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data validator.
        
        Args:
            config: Validation configuration dictionary
        """
        self.config = config or {}
        self.validation_results: List[Dict[str, Any]] = []
    
    def validate_data_types(self, df: pd.DataFrame, schema: Dict[str, str]) -> bool:
        """
        Validate that columns match expected data types.
        
        Args:
            df: Input DataFrame
            schema: Dictionary mapping column names to expected dtypes
        
        Returns:
            True if all validations pass
        
        Raises:
            ValidationError: If validation fails
        """
        errors = []
        passed_columns = []
        detailed_results = []
        
        for column, expected_dtype in schema.items():
            if column not in df.columns:
                error_msg = f"Column '{column}' not found in DataFrame"
                errors.append(error_msg)
                detailed_results.append({
                    'column': column,
                    'expected': expected_dtype,
                    'found': 'MISSING',
                    'action': 'validation failed'
                })
                logger.error(f"Column: {column} | Expected: {expected_dtype} | Found: MISSING | Action: validation failed")
                continue
            
            actual_dtype = str(df[column].dtype)
            expected_dtype_str = str(expected_dtype)
            
            # Flexible type checking
            if not self._dtype_matches(actual_dtype, expected_dtype_str):
                # Check if coercion is possible
                can_coerce = False
                suggested_action = 'validation failed'
                
                # Check if values can be coerced
                try:
                    if 'int' in expected_dtype_str.lower() or 'float' in expected_dtype_str.lower():
                        test_coerce = pd.to_numeric(df[column], errors='coerce')
                        if test_coerce.notna().sum() > len(df) * 0.8:  # 80% can be coerced
                            can_coerce = True
                            suggested_action = 'can be coerced to numeric'
                    elif 'bool' in expected_dtype_str.lower():
                        can_coerce = True
                        suggested_action = 'can be coerced to bool'
                except:
                    pass
                
                error_msg = f"Column '{column}' has dtype '{actual_dtype}', expected '{expected_dtype_str}'"
                errors.append(error_msg)
                detailed_results.append({
                    'column': column,
                    'expected': expected_dtype_str,
                    'found': actual_dtype,
                    'action': suggested_action
                })
                logger.error(f"Column: {column} | Expected: {expected_dtype_str} | Found: {actual_dtype} | Action: {suggested_action}")
            else:
                passed_columns.append(column)
                detailed_results.append({
                    'column': column,
                    'expected': expected_dtype_str,
                    'found': actual_dtype,
                    'action': 'passed'
                })
                logger.debug(f"Column: {column} | Expected: {expected_dtype_str} | Found: {actual_dtype} | Action: passed")
        
        if errors:
            error_msg = "Data type validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(f"Data type validation failed: {len(errors)} column(s) failed, {len(passed_columns)} column(s) passed")
            self.validation_results.append({
                'check': 'data_types',
                'status': 'failed',
                'errors': errors,
                'passed_count': len(passed_columns),
                'failed_count': len(errors),
                'detailed_results': detailed_results
            })
            raise ValidationError(error_msg)
        
        logger.info(f"Data type validation passed: {len(passed_columns)} column(s) validated successfully")
        self.validation_results.append({
            'check': 'data_types',
            'status': 'passed',
            'validated_columns': len(passed_columns),
            'detailed_results': detailed_results
        })
        return True
    
    def _dtype_matches(self, actual: str, expected: str) -> bool:
        """Check if actual dtype matches expected (with flexibility)."""
        dtype_mapping = {
            'int': ['int64', 'int32', 'int16', 'int8', 'Int64', 'Int32'],
            'float': ['float64', 'float32', 'float16'],
            'object': ['object', 'string'],
            'datetime': ['datetime64[ns]', 'datetime64[us]'],
            'bool': ['bool', 'boolean']
        }
        
        # Exact match
        if actual == expected:
            return True
        
        # Flexible match
        for base_type, variants in dtype_mapping.items():
            if base_type in expected.lower() and actual in variants:
                return True
        
        return False
    
    def validate_value_ranges(self, df: pd.DataFrame, ranges: Dict[str, Dict[str, float]]) -> bool:
        """
        Validate that column values fall within specified ranges.
        
        Args:
            df: Input DataFrame
            ranges: Dictionary mapping column names to min/max values
                   e.g., {'age': {'min': 0, 'max': 120}}
        
        Returns:
            True if all validations pass
        
        Raises:
            ValidationError: If validation fails
        """
        errors = []
        passed_columns = []
        
        for column, range_config in ranges.items():
            if column not in df.columns:
                error_msg = f"Column '{column}' not found in DataFrame"
                errors.append(error_msg)
                logger.error(f"Column: {column} | Range check: FAILED | Reason: column not found")
                continue
            
            min_val = range_config.get('min')
            max_val = range_config.get('max')
            column_errors = []
            
            if min_val is not None:
                invalid_min = df[column] < min_val
                if invalid_min.any():
                    count = invalid_min.sum()
                    error_msg = f"Column '{column}': {count} values below minimum {min_val}"
                    column_errors.append(error_msg)
                    logger.error(f"Column: {column} | Range: [{min_val}, {max_val}] | Found: {count} values < {min_val}")
            
            if max_val is not None:
                invalid_max = df[column] > max_val
                if invalid_max.any():
                    count = invalid_max.sum()
                    error_msg = f"Column '{column}': {count} values above maximum {max_val}"
                    column_errors.append(error_msg)
                    logger.error(f"Column: {column} | Range: [{min_val}, {max_val}] | Found: {count} values > {max_val}")
            
            if column_errors:
                errors.extend(column_errors)
            else:
                passed_columns.append(column)
                logger.debug(f"Column: {column} | Range: [{min_val}, {max_val}] | Action: passed")
        
        if errors:
            error_msg = "Value range validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(f"Value range validation failed: {len(errors)} issue(s) found, {len(passed_columns)} column(s) passed")
            self.validation_results.append({
                'check': 'value_ranges',
                'status': 'failed',
                'errors': errors,
                'passed_count': len(passed_columns),
                'failed_count': len(errors)
            })
            raise ValidationError(error_msg)
        
        logger.info(f"Value range validation passed: {len(passed_columns)} column(s) validated successfully")
        self.validation_results.append({
            'check': 'value_ranges',
            'status': 'passed',
            'validated_columns': len(passed_columns)
        })
        return True
    
    def validate_not_null(self, df: pd.DataFrame, columns: List[str]) -> bool:
        """
        Validate that specified columns contain no null values.
        
        Args:
            df: Input DataFrame
            columns: List of columns that must not be null
        
        Returns:
            True if all validations pass
        
        Raises:
            ValidationError: If validation fails
        """
        errors = []
        passed_columns = []
        
        for column in columns:
            if column not in df.columns:
                error_msg = f"Column '{column}' not found in DataFrame"
                errors.append(error_msg)
                logger.error(f"Column: {column} | Not-null check: FAILED | Reason: column not found")
                continue
            
            null_count = df[column].isnull().sum()
            if null_count > 0:
                error_msg = f"Column '{column}' contains {null_count} null values"
                errors.append(error_msg)
                logger.error(f"Column: {column} | Not-null check: FAILED | Found: {null_count} null values")
            else:
                passed_columns.append(column)
                logger.debug(f"Column: {column} | Not-null check: PASSED")
        
        if errors:
            error_msg = "Not-null validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(f"Not-null validation failed: {len(errors)} column(s) failed, {len(passed_columns)} column(s) passed")
            self.validation_results.append({
                'check': 'not_null',
                'status': 'failed',
                'errors': errors,
                'passed_count': len(passed_columns),
                'failed_count': len(errors)
            })
            raise ValidationError(error_msg)
        
        logger.info(f"Not-null validation passed: {len(passed_columns)} column(s) validated successfully")
        self.validation_results.append({
            'check': 'not_null',
            'status': 'passed',
            'validated_columns': len(passed_columns)
        })
        return True
    
    def validate_unique(self, df: pd.DataFrame, columns: List[str]) -> bool:
        """
        Validate that specified columns contain unique values.
        
        Args:
            df: Input DataFrame
            columns: List of columns that must be unique
        
        Returns:
            True if all validations pass
        
        Raises:
            ValidationError: If validation fails
        """
        errors = []
        
        for column in columns:
            if column not in df.columns:
                errors.append(f"Column '{column}' not found in DataFrame")
                continue
            
            duplicate_count = df[column].duplicated().sum()
            if duplicate_count > 0:
                errors.append(f"Column '{column}' contains {duplicate_count} duplicate values")
        
        if errors:
            error_msg = "Unique validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            self.validation_results.append({
                'check': 'unique',
                'status': 'failed',
                'errors': errors
            })
            raise ValidationError(error_msg)
        
        logger.info("Unique validation passed")
        self.validation_results.append({'check': 'unique', 'status': 'passed'})
        return True
    
    def validate_custom(self, df: pd.DataFrame, validations: List[Dict[str, Any]]) -> bool:
        """
        Run custom validation functions.
        
        Args:
            df: Input DataFrame
            validations: List of validation dictionaries with 'name', 'function', and optional 'args'
        
        Returns:
            True if all validations pass
        
        Raises:
            ValidationError: If validation fails
        """
        errors = []
        
        for validation in validations:
            name = validation.get('name', 'unknown')
            func = validation.get('function')
            args = validation.get('args', {})
            
            if not callable(func):
                errors.append(f"Custom validation '{name}': function is not callable")
                continue
            
            try:
                result = func(df, **args)
                if not result:
                    errors.append(f"Custom validation '{name}' failed")
            except Exception as e:
                errors.append(f"Custom validation '{name}' raised exception: {str(e)}")
        
        if errors:
            error_msg = "Custom validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            self.validation_results.append({
                'check': 'custom',
                'status': 'failed',
                'errors': errors
            })
            raise ValidationError(error_msg)
        
        logger.info("Custom validation passed")
        self.validation_results.append({'check': 'custom', 'status': 'passed'})
        return True
    
    def validate(self, df: pd.DataFrame, validation_config: Dict[str, Any]) -> bool:
        """
        Run all validation checks based on configuration.
        
        Args:
            df: Input DataFrame
            validation_config: Configuration dictionary with validation rules
        
        Returns:
            True if all validations pass
        
        Raises:
            ValidationError: If any validation fails
        """
        logger.info("Starting data validation process")
        self.validation_results = []
        
        # Data type validation
        if validation_config.get('data_types', {}).get('enabled', False):
            schema = validation_config['data_types'].get('schema', {})
            self.validate_data_types(df, schema)
        
        # Value range validation
        if validation_config.get('value_ranges', {}).get('enabled', False):
            ranges = validation_config['value_ranges'].get('ranges', {})
            self.validate_value_ranges(df, ranges)
        
        # Not-null validation
        if validation_config.get('not_null', {}).get('enabled', False):
            columns = validation_config['not_null'].get('columns', [])
            self.validate_not_null(df, columns)
        
        # Unique validation
        if validation_config.get('unique', {}).get('enabled', False):
            columns = validation_config['unique'].get('columns', [])
            self.validate_unique(df, columns)
        
        # Custom validation
        if validation_config.get('custom', {}).get('enabled', False):
            validations = validation_config['custom'].get('validations', [])
            self.validate_custom(df, validations)
        
        logger.info("Data validation process completed successfully")
        return True
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get validation results."""
        return self.validation_results
