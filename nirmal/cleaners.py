"""
Data cleaning modules for various data quality issues.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional


logger = logging.getLogger(__name__)


class DataCleaner:
    """Data cleaning operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data cleaner.
        
        Args:
            config: Cleaning configuration dictionary
        """
        self.config = config or {}
    
    def clean_numeric_columns(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Clean numeric columns by converting invalid values (non-numeric strings) to NaN.
        
        Args:
            df: Input DataFrame
            columns: Columns to clean (None = auto-detect numeric columns)
        
        Returns:
            DataFrame with numeric columns cleaned
        """
        df_cleaned = df.copy()
        columns_modified = []
        total_coercions = 0
        
        if columns is None:
            # Auto-detect columns that should be numeric
            numeric_columns = []
            for col in df_cleaned.columns:
                # Try to convert to numeric, if many succeed, consider it numeric
                numeric_test = pd.to_numeric(df_cleaned[col], errors='coerce')
                if numeric_test.notna().sum() > len(df_cleaned) * 0.5:  # More than 50% numeric
                    numeric_columns.append(col)
        else:
            numeric_columns = [col for col in columns if col in df_cleaned.columns]
        
        for column in numeric_columns:
            original_dtype = str(df_cleaned[column].dtype)
            # Convert to numeric, invalid values become NaN
            numeric_series = pd.to_numeric(df_cleaned[column], errors='coerce')
            
            # Count type coercions (non-numeric strings converted to NaN)
            # Count values that were not NaN before but are NaN after
            was_not_na = df_cleaned[column].notna()
            is_now_na = numeric_series.isna()
            coercion_count = (was_not_na & is_now_na).sum()
            
            # Handle negative values for columns that should be positive
            # Common columns that should never be negative: salary, age, price, count, etc.
            negative_sensitive_keywords = ['salary', 'age', 'price', 'cost', 'amount', 'count', 'quantity', 'revenue', 'income']
            # Ensure column name is a string (pandas column names can be various types)
            column_str = str(column) if not isinstance(column, str) else column
            column_lower = column_str.lower()
            has_negative_sensitive_keyword = any(keyword in column_lower for keyword in negative_sensitive_keywords)
            
            if has_negative_sensitive_keyword:
                negative_mask = (numeric_series < 0) & numeric_series.notna()
                negative_count = negative_mask.sum()
                if negative_count > 0:
                    logger.warning(f"Column '{column}': Found {negative_count} negative values (invalid for this column type). Converting to NaN.")
                    numeric_series.loc[negative_mask] = np.nan
                    coercion_count += negative_count
            
            # Only update if there are changes
            if not numeric_series.equals(df_cleaned[column]):
                df_cleaned[column] = numeric_series
                new_dtype = str(numeric_series.dtype)
                columns_modified.append(column)
                total_coercions += coercion_count
                
                if coercion_count > 0:
                    logger.info(f"Column '{column}': converted {coercion_count} invalid values to NaN (dtype: {original_dtype} -> {new_dtype})")
                elif original_dtype != new_dtype:
                    logger.info(f"Column '{column}': dtype converted from {original_dtype} to {new_dtype}")
        
        if columns_modified:
            logger.info(f"Type coercion: {total_coercions} values coerced across {len(columns_modified)} columns: {columns_modified}")
        
        return df_cleaned
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None, keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from DataFrame.
        
        Args:
            df: Input DataFrame
            subset: Column names to consider for duplicates (None = auto-exclude ID columns, [] = all columns)
            keep: Which duplicates to keep ('first', 'last', False)
        
        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(df)
        
        # If subset is None, auto-exclude common ID columns
        if subset is None:
            # Common ID column patterns to exclude
            id_patterns = ['id', 'ID', 'Id', 'uuid', 'UUID', 'guid', 'GUID']
            id_columns = []
            
            for col in df.columns:
                # Exact match for common ID column names
                if col in id_patterns:
                    id_columns.append(col)
                # Pattern match: ends with '_id' or '_ID'
                elif str(col).lower().endswith('_id'):
                    id_columns.append(col)
            
            # Use all columns except ID columns
            if id_columns:
                subset = [col for col in df.columns if col not in id_columns]
                logger.info(f"Auto-excluded ID columns from duplicate check: {id_columns}")
                logger.info(f"Checking duplicates based on: {subset}")
            else:
                # No ID columns found, use all columns
                subset = None
        
        # Before removing duplicates, ensure text columns are trimmed for accurate comparison
        # This helps catch duplicates like "Alice" vs " Alice " or "Alice" vs "Alice  "
        text_cols = df.select_dtypes(include=['object']).columns
        df_for_dup_check = df.copy()
        for col in text_cols:
            if col in df_for_dup_check.columns:
                df_for_dup_check[col] = df_for_dup_check[col].astype(str).str.strip()
                # Convert 'nan' strings back to NaN for proper comparison
                df_for_dup_check[col] = df_for_dup_check[col].replace('nan', np.nan)
        
        df_cleaned = df_for_dup_check.drop_duplicates(subset=subset, keep=keep)
        removed_count = initial_count - len(df_cleaned)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate rows")
        else:
            logger.info("No duplicate rows found")
        return df_cleaned
    
    def handle_invalid_zeros(self, df: pd.DataFrame, domain_aware_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Handle invalid zeros (masked missing data) using domain-aware strategies.
        
        Args:
            df: Input DataFrame
            domain_aware_config: Configuration for domain-aware handling
                Example: {
                    'lat': {'strategy': 'neighborhood_mean', 'group_by': 'neighbourhood'},
                    'long': {'strategy': 'neighborhood_mean', 'group_by': 'neighbourhood'},
                    'construction_year': {'strategy': 'median'},
                    'flag_invalid': True  # Add flag columns
                }
        
        Returns:
            DataFrame with invalid zeros handled
        """
        df_cleaned = df.copy()
        if not domain_aware_config:
            return df_cleaned
        
        flag_invalid = domain_aware_config.get('flag_invalid', False)
        columns_modified = []
        rows_dropped = 0
        initial_count = len(df_cleaned)
        
        for col_name, col_config in domain_aware_config.items():
            if col_name == 'flag_invalid' or col_name not in df_cleaned.columns:
                continue
            
            strategy = col_config.get('strategy', 'drop')
            group_by = col_config.get('group_by')
            
            # Convert to numeric
            numeric_series = pd.to_numeric(df_cleaned[col_name], errors='coerce')
            
            # Find invalid zeros (where value is exactly 0 and should not be)
            invalid_mask = (numeric_series == 0) & numeric_series.notna()
            invalid_count = invalid_mask.sum()
            
            if invalid_count == 0:
                continue
            
            columns_modified.append(col_name)
            logger.info(f"Found {invalid_count} invalid zeros in column '{col_name}'")
            
            # Add flag column if requested
            if flag_invalid:
                flag_col_name = f'is_valid_{col_name}'
                df_cleaned[flag_col_name] = ~invalid_mask
                logger.info(f"Added flag column '{flag_col_name}' for invalid values")
            
            # Apply strategy
            if strategy == 'drop':
                df_cleaned = df_cleaned[~invalid_mask]
                rows_dropped += invalid_count
                logger.info(f"Dropped {invalid_count} rows with invalid zeros in '{col_name}'")
            
            elif strategy == 'neighborhood_mean' and group_by:
                if group_by in df_cleaned.columns:
                    # Calculate mean by group, excluding zeros
                    valid_data = df_cleaned[~invalid_mask]
                    group_means = valid_data.groupby(group_by)[col_name].mean()
                    
                    # Fill invalid zeros with group mean
                    for idx in df_cleaned[invalid_mask].index:
                        group_val = df_cleaned.loc[idx, group_by]
                        if group_val in group_means.index and pd.notna(group_means[group_val]):
                            df_cleaned.loc[idx, col_name] = group_means[group_val]
                        else:
                            # If group not found, use overall mean
                            overall_mean = valid_data[col_name].mean()
                            if pd.notna(overall_mean):
                                df_cleaned.loc[idx, col_name] = overall_mean
                            else:
                                df_cleaned.loc[idx, col_name] = np.nan
                    
                    filled_count = invalid_mask.sum() - df_cleaned.loc[invalid_mask, col_name].isna().sum()
                    logger.info(f"Filled {filled_count} invalid zeros in '{col_name}' using {group_by} mean")
                else:
                    logger.warning(f"Group column '{group_by}' not found, using overall mean for '{col_name}'")
                    overall_mean = df_cleaned[~invalid_mask][col_name].mean()
                    if pd.notna(overall_mean):
                        df_cleaned.loc[invalid_mask, col_name] = overall_mean
                        logger.info(f"Filled {invalid_count} invalid zeros in '{col_name}' using overall mean")
            
            elif strategy == 'median':
                median_val = df_cleaned[~invalid_mask][col_name].median()
                if pd.notna(median_val):
                    df_cleaned.loc[invalid_mask, col_name] = median_val
                    logger.info(f"Filled {invalid_count} invalid zeros in '{col_name}' using median ({median_val})")
                else:
                    logger.warning(f"Cannot compute median for '{col_name}', converting to NaN")
                    df_cleaned.loc[invalid_mask, col_name] = np.nan
            
            elif strategy == 'mode':
                mode_val = df_cleaned[~invalid_mask][col_name].mode()
                if len(mode_val) > 0 and pd.notna(mode_val.iloc[0]):
                    df_cleaned.loc[invalid_mask, col_name] = mode_val.iloc[0]
                    logger.info(f"Filled {invalid_count} invalid zeros in '{col_name}' using mode ({mode_val.iloc[0]})")
                else:
                    logger.warning(f"Cannot compute mode for '{col_name}', converting to NaN")
                    df_cleaned.loc[invalid_mask, col_name] = np.nan
            
            else:
                # Default: convert to NaN
                df_cleaned.loc[invalid_mask, col_name] = np.nan
                logger.info(f"Converted {invalid_count} invalid zeros in '{col_name}' to NaN")
        
        final_count = len(df_cleaned)
        if columns_modified:
            logger.info(f"Invalid zeros handling: {initial_count} -> {final_count} rows, modified columns: {columns_modified}")
        if rows_dropped > 0:
            logger.info(f"Dropped {rows_dropped} rows due to invalid zeros")
        
        return df_cleaned
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop', fill_value: Any = None, 
                             columns: Optional[List[str]] = None, domain_aware_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.
        Uses appropriate fill strategies for numeric vs text columns.
        
        Args:
            df: Input DataFrame
            strategy: Strategy to use ('drop', 'fill', 'forward_fill', 'backward_fill')
            fill_value: Value to use for filling (if strategy is 'fill', None = auto-detect)
            columns: Specific columns to process (None = all columns)
            domain_aware_config: Configuration for domain-aware handling of specific columns
        
        Returns:
            DataFrame with missing values handled
        """
        df_cleaned = df.copy()
        columns_to_process = list(columns) if columns else list(df.columns)
        
        initial_missing = df_cleaned[columns_to_process].isnull().sum().sum()
        initial_count = len(df_cleaned)
        type_coercion_count = 0
        
        if strategy == 'drop':
            df_cleaned = df_cleaned.dropna(subset=columns_to_process)
            rows_dropped = initial_count - len(df_cleaned)
            logger.info(f"Dropped {rows_dropped} rows with missing values in specified columns")
        
        elif strategy == 'fill':
            # Separate numeric and text columns
            numeric_cols = []
            text_cols = []
            
            for col in columns_to_process:
                if col not in df_cleaned.columns:
                    continue
                
                # Check if column is numeric or should be numeric
                is_numeric = pd.api.types.is_numeric_dtype(df_cleaned[col])
                
                # If not numeric but contains mostly numbers, try to convert
                if not is_numeric:
                    # Try to convert to numeric to see if it's actually numeric data
                    numeric_test = pd.to_numeric(df_cleaned[col], errors='coerce')
                    non_null_count = numeric_test.notna().sum()
                    total_count = len(df_cleaned[col])
                    # If more than 50% can be converted to numeric, treat as numeric
                    if total_count > 0 and (non_null_count / total_count) > 0.5:
                        df_cleaned[col] = numeric_test
                        is_numeric = True
                        logger.info(f"Converted column '{col}' to numeric type for missing value handling")
                
                if is_numeric:
                    numeric_cols.append(col)
                else:
                    text_cols.append(col)
            
            # Fill numeric columns
            if numeric_cols:
                if fill_value is not None and pd.api.types.is_number(fill_value):
                    # Use provided numeric fill value
                    filled_count = df_cleaned[numeric_cols].isnull().sum().sum()
                    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(fill_value)
                    logger.info(f"Filled {filled_count} missing values in numeric columns with {fill_value}")
                else:
                    # Auto-detect: use median for numeric columns
                    filled_by_col = {}
                    for col in numeric_cols:
                        missing_count = df_cleaned[col].isnull().sum()
                        if missing_count > 0:
                            median_val = df_cleaned[col].median()
                            if pd.notna(median_val):
                                df_cleaned[col] = df_cleaned[col].fillna(median_val)
                                filled_by_col[col] = (missing_count, median_val)
                            else:
                                # Only use 0 as last resort, but log warning
                                df_cleaned[col] = df_cleaned[col].fillna(0)
                                filled_by_col[col] = (missing_count, 0)
                                logger.warning(f"Column '{col}': median is NaN, filled with 0 (may be invalid)")
                    
                    if filled_by_col:
                        for col, (count, val) in filled_by_col.items():
                            logger.info(f"Filled {count} missing values in '{col}' with {val}")
            
            # Fill text columns
            if text_cols:
                if fill_value is not None and isinstance(fill_value, str):
                    # Use provided text fill value
                    filled_count = df_cleaned[text_cols].isnull().sum().sum()
                    df_cleaned[text_cols] = df_cleaned[text_cols].fillna(fill_value)
                    logger.info(f"Filled {filled_count} missing values in text columns with '{fill_value}'")
                else:
                    # Auto-detect: use mode or "Unknown"
                    for col in text_cols:
                        missing_count = df_cleaned[col].isnull().sum()
                        if missing_count > 0:
                            mode_val = df_cleaned[col].mode()
                            if len(mode_val) > 0 and pd.notna(mode_val.iloc[0]):
                                df_cleaned[col] = df_cleaned[col].fillna(mode_val.iloc[0])
                                logger.info(f"Filled {missing_count} missing values in '{col}' with mode '{mode_val.iloc[0]}'")
                            else:
                                df_cleaned[col] = df_cleaned[col].fillna("Unknown")
                                logger.info(f"Filled {missing_count} missing values in '{col}' with 'Unknown'")
        
        elif strategy == 'forward_fill':
            filled_count = df_cleaned[columns_to_process].isnull().sum().sum()
            df_cleaned[columns_to_process] = df_cleaned[columns_to_process].ffill()
            logger.info(f"Forward filled {filled_count} missing values")
        
        elif strategy == 'backward_fill':
            filled_count = df_cleaned[columns_to_process].isnull().sum().sum()
            df_cleaned[columns_to_process] = df_cleaned[columns_to_process].bfill()
            logger.info(f"Backward filled {filled_count} missing values")
        
        remaining_missing = df_cleaned[columns_to_process].isnull().sum().sum()
        rows_dropped = initial_count - len(df_cleaned)
        logger.info(f"Missing values handling: {initial_missing} -> {remaining_missing} missing, {rows_dropped} rows dropped")
        
        return df_cleaned
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr', 
                       factor: float = 1.5, cap: bool = False) -> pd.DataFrame:
        """
        Remove or cap outliers from specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to process
            method: Method to use ('iqr', 'zscore')
            factor: Factor for IQR method (default: 1.5)
            cap: If True, cap outliers instead of removing rows (default: False)
        
        Returns:
            DataFrame with outliers removed or capped
        """
        df_cleaned = df.copy()
        initial_count = len(df_cleaned)
        columns_modified = []
        total_outliers_handled = 0
        
        for column in columns:
            if column not in df_cleaned.columns:
                logger.warning(f"Column '{column}' not found, skipping")
                continue
            
            # Convert column to numeric, coercing errors to NaN
            numeric_series = pd.to_numeric(df_cleaned[column], errors='coerce')
            
            # Check if column has any numeric values
            if numeric_series.isna().all():
                logger.warning(f"Column '{column}' cannot be converted to numeric, skipping outlier treatment")
                continue
            
            # Check if column is numeric enough for outlier detection
            numeric_count = numeric_series.notna().sum()
            if numeric_count < 2:  # Need at least 2 values for statistics
                logger.warning(f"Column '{column}' has insufficient numeric values ({numeric_count}), skipping outlier treatment")
                continue
            
            try:
                if method == 'iqr':
                    # Use only numeric values for quantile calculation
                    Q1 = numeric_series.quantile(0.25)
                    Q3 = numeric_series.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if pd.isna(IQR) or IQR == 0:
                        logger.warning(f"Column '{column}' has IQR of {IQR}, skipping outlier treatment")
                        continue
                    
                    lower_bound = Q1 - factor * IQR
                    upper_bound = Q3 + factor * IQR
                    
                    if cap:
                        # Cap outliers instead of removing
                        outliers_low = (numeric_series < lower_bound) & numeric_series.notna()
                        outliers_high = (numeric_series > upper_bound) & numeric_series.notna()
                        outliers_count = outliers_low.sum() + outliers_high.sum()
                        
                        if outliers_count > 0:
                            df_cleaned.loc[outliers_low, column] = lower_bound
                            df_cleaned.loc[outliers_high, column] = upper_bound
                            columns_modified.append(column)
                            total_outliers_handled += outliers_count
                            logger.info(f"Capped {outliers_count} outliers in column '{column}' (low: {outliers_low.sum()}, high: {outliers_high.sum()})")
                    else:
                        # Remove outliers
                        mask = numeric_series.isna() | ((numeric_series >= lower_bound) & (numeric_series <= upper_bound))
                        outliers_removed = (~mask).sum()
                        if outliers_removed > 0:
                            df_cleaned = df_cleaned[mask]
                            total_outliers_handled += outliers_removed
                            logger.info(f"Removed {outliers_removed} outliers from column '{column}'")
                
                elif method == 'zscore':
                    # Calculate z-scores only on numeric values
                    mean_val = numeric_series.mean()
                    std_val = numeric_series.std()
                    
                    if pd.isna(std_val) or std_val == 0:
                        logger.warning(f"Column '{column}' has std of {std_val}, skipping outlier treatment")
                        continue
                    
                    z_scores = np.abs((numeric_series - mean_val) / std_val)
                    z_threshold = 3  # Standard threshold for z-score
                    
                    if cap:
                        # Cap outliers instead of removing
                        outliers_mask = (z_scores >= z_threshold) & numeric_series.notna()
                        outliers_count = outliers_mask.sum()
                        
                        if outliers_count > 0:
                            # Cap to mean ± 3*std
                            lower_bound = mean_val - z_threshold * std_val
                            upper_bound = mean_val + z_threshold * std_val
                            df_cleaned.loc[outliers_mask & (numeric_series < mean_val), column] = lower_bound
                            df_cleaned.loc[outliers_mask & (numeric_series > mean_val), column] = upper_bound
                            columns_modified.append(column)
                            total_outliers_handled += outliers_count
                            logger.info(f"Capped {outliers_count} outliers in column '{column}' using z-score method")
                    else:
                        # Remove outliers
                        mask = numeric_series.isna() | (z_scores < z_threshold)
                        outliers_removed = (~mask).sum()
                        if outliers_removed > 0:
                            df_cleaned = df_cleaned[mask]
                            total_outliers_handled += outliers_removed
                            logger.info(f"Removed {outliers_removed} outliers from column '{column}' using z-score method")
                
                else:
                    logger.warning(f"Unknown outlier method '{method}', skipping")
                    continue
                
            except Exception as e:
                logger.warning(f"Error processing outliers for column '{column}': {e}, skipping")
                continue
        
        final_count = len(df_cleaned)
        action = "capped" if cap else "removed"
        if total_outliers_handled > 0:
            logger.info(f"Outlier treatment ({action}): {initial_count} -> {final_count} rows, {total_outliers_handled} outliers {action}, modified columns: {columns_modified}")
        else:
            logger.info(f"No outliers found or {action}")
        return df_cleaned
    
    def standardize_text(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Standardize text columns (trim whitespace, convert to lowercase).
        
        Args:
            df: Input DataFrame
            columns: Text columns to standardize
        
        Returns:
            DataFrame with standardized text
        """
        df_cleaned = df.copy()
        
        for column in columns:
            if column not in df_cleaned.columns:
                logger.warning(f"Column '{column}' not found, skipping")
                continue
            
            if df_cleaned[column].dtype == 'object':
                df_cleaned[column] = df_cleaned[column].astype(str).str.strip().str.lower()
                logger.info(f"Standardized text in column '{column}'")
        
        return df_cleaned
    
    def clean(self, df: pd.DataFrame, cleaning_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply all cleaning operations based on configuration.
        
        Args:
            df: Input DataFrame
            cleaning_config: Configuration dictionary with cleaning steps
        
        Returns:
            Cleaned DataFrame
        """
        df_cleaned = df.copy()
        initial_count = len(df_cleaned)
        logger.info(f"Starting data cleaning process: {initial_count} rows, {len(df_cleaned.columns)} columns")
        
        # Step 0.5: Convert empty strings to NaN (important for CSV files)
        # This ensures empty cells are treated as missing values
        empty_strings_before = (df_cleaned == '').sum().sum()
        if empty_strings_before > 0:
            # Replace empty strings with NaN for all columns
            df_cleaned = df_cleaned.replace('', np.nan)
            # Also handle whitespace-only strings
            for col in df_cleaned.columns:
                if df_cleaned[col].dtype == 'object':
                    df_cleaned[col] = df_cleaned[col].replace(r'^\s*$', np.nan, regex=True)
            logger.info(f"Converted {empty_strings_before} empty strings to NaN")
        
        # Step 0.6: Remove completely empty rows (all NaN)
        rows_before_empty = len(df_cleaned)
        df_cleaned = df_cleaned.dropna(how='all')
        rows_dropped_empty = rows_before_empty - len(df_cleaned)
        if rows_dropped_empty > 0:
            logger.info(f"Dropped {rows_dropped_empty} completely empty rows")
        
        # Step 0.7: Trim whitespace from text columns (always do this for better duplicate detection)
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                # Only process non-null values to avoid converting NaN to 'nan' string
                mask = df_cleaned[col].notna()
                if mask.any():
                    df_cleaned.loc[mask, col] = df_cleaned.loc[mask, col].astype(str).str.strip()
                    # Convert empty strings back to NaN
                    df_cleaned.loc[df_cleaned[col] == '', col] = np.nan
        
        # Step 0: Handle invalid zeros (domain-aware handling)
        if cleaning_config.get('handle_invalid_zeros', {}).get('enabled', False):
            invalid_zeros_config = cleaning_config['handle_invalid_zeros']
            domain_config = {k: v for k, v in invalid_zeros_config.items() if k != 'enabled'}
            df_cleaned = self.handle_invalid_zeros(df_cleaned, domain_config)
            logger.info(f"After invalid zeros handling: {len(df_cleaned)} rows")
        
        # Step 1: Clean numeric columns (convert invalid values like 'abc' to NaN)
        # This runs by default to handle mixed-type columns
        if cleaning_config.get('clean_numeric', {}).get('enabled', True):
            numeric_config = cleaning_config.get('clean_numeric', {})
            df_cleaned = self.clean_numeric_columns(
                df_cleaned,
                columns=numeric_config.get('columns')  # None = auto-detect
            )
        
        # Step 2: Remove duplicates
        remove_duplicates_config = cleaning_config.get('remove_duplicates', {})
        # Handle both boolean and dict formats (safety check - should be normalized by engine)
        if isinstance(remove_duplicates_config, bool):
            remove_duplicates_enabled = remove_duplicates_config
            dup_config = {}
        elif isinstance(remove_duplicates_config, dict):
            remove_duplicates_enabled = remove_duplicates_config.get('enabled', False)
            dup_config = remove_duplicates_config
        else:
            remove_duplicates_enabled = False
            dup_config = {}
        
        if remove_duplicates_enabled:
            rows_before = len(df_cleaned)
            df_cleaned = self.remove_duplicates(
                df_cleaned,
                subset=dup_config.get('subset'),
                keep=dup_config.get('keep', 'first')
            )
            rows_dropped = rows_before - len(df_cleaned)
            if rows_dropped > 0:
                logger.info(f"Dropped {rows_dropped} duplicate rows (step: remove_duplicates)")
        
        # Step 3: Handle missing values (after numeric cleaning)
        handle_missing_config = cleaning_config.get('handle_missing', {})
        # Handle both boolean and dict formats (safety check - should be normalized by engine)
        if isinstance(handle_missing_config, bool):
            handle_missing_enabled = handle_missing_config
            missing_config = {}
        elif isinstance(handle_missing_config, dict):
            handle_missing_enabled = handle_missing_config.get('enabled', False)
            missing_config = handle_missing_config
        else:
            handle_missing_enabled = False
            missing_config = {}
        
        if handle_missing_enabled:
            rows_before = len(df_cleaned)
            missing_before = df_cleaned.isnull().sum().sum()
            strategy = missing_config.get('strategy', 'drop')
            logger.info(f"Handling missing values: {missing_before} missing values found, strategy: {strategy}")
            
            df_cleaned = self.handle_missing_values(
                df_cleaned,
                strategy=strategy,
                fill_value=missing_config.get('fill_value'),
                columns=missing_config.get('columns'),
                domain_aware_config=missing_config.get('domain_aware')
            )
            
            missing_after = df_cleaned.isnull().sum().sum()
            rows_dropped = rows_before - len(df_cleaned)
            logger.info(f"After handling missing values: {missing_after} missing values remaining, {rows_dropped} rows dropped")
        else:
            logger.info("handle_missing is disabled or not configured")
        
        # Step 4: Handle outliers (new config with auto-detection)
        if cleaning_config.get('outliers', {}).get('enabled', False):
            outlier_config = cleaning_config['outliers']
            rows_before = len(df_cleaned)
            
            # Auto-detect numeric columns if columns list is empty
            columns_to_process = outlier_config.get('columns', [])
            if not columns_to_process:
                # Auto-detect all numeric columns
                numeric_cols = df_cleaned.select_dtypes(include=['number']).columns.tolist()
                columns_to_process = numeric_cols
                logger.info(f"Auto-detected {len(columns_to_process)} numeric columns for outlier handling: {columns_to_process}")
            
            if columns_to_process:
                df_cleaned = self.remove_outliers(
                    df_cleaned,
                    columns=columns_to_process,
                    method=outlier_config.get('method', 'iqr'),
                    factor=outlier_config.get('factor', 1.5),
                    cap=outlier_config.get('cap', True)
                )
                rows_dropped = rows_before - len(df_cleaned)
                if rows_dropped > 0:
                    logger.info(f"Dropped {rows_dropped} rows (step: outliers)")
        
        # Step 4b: Remove or cap outliers (legacy config)
        elif cleaning_config.get('remove_outliers', {}).get('enabled', False):
            outlier_config = cleaning_config['remove_outliers']
            rows_before = len(df_cleaned)
            df_cleaned = self.remove_outliers(
                df_cleaned,
                columns=outlier_config.get('columns', []),
                method=outlier_config.get('method', 'iqr'),
                factor=outlier_config.get('factor', 1.5),
                cap=outlier_config.get('cap', False)
            )
            rows_dropped = rows_before - len(df_cleaned)
            if rows_dropped > 0:
                logger.info(f"Dropped {rows_dropped} rows (step: remove_outliers)")
        
        # Step 5: Standardize text
        if cleaning_config.get('standardize_text', {}).get('enabled', False):
            text_config = cleaning_config['standardize_text']
            columns_before = set(df_cleaned.columns)
            df_cleaned = self.standardize_text(
                df_cleaned,
                columns=text_config.get('columns', [])
            )
            columns_modified = [col for col in text_config.get('columns', []) if col in columns_before]
            if columns_modified:
                logger.info(f"Modified {len(columns_modified)} text columns: {columns_modified}")
        
        final_count = len(df_cleaned)
        total_rows_dropped = initial_count - final_count
        logger.info(f"Data cleaning process completed: {initial_count} -> {final_count} rows ({total_rows_dropped} dropped), {len(df_cleaned.columns)} columns")
        return df_cleaned
