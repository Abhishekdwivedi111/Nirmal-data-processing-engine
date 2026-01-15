"""
Data preprocessing modules for feature engineering and transformation.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessing operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data preprocessor.
        
        Args:
            config: Preprocessing configuration dictionary
        """
        self.config = config or {}
        self.scalers = {}
        self.encoders = {}
    
    def normalize(self, df: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
        """
        Normalize numeric columns using scaling.
        
        Args:
            df: Input DataFrame
            columns: Columns to normalize
            method: Normalization method ('standard', 'minmax')
        
        Returns:
            DataFrame with normalized columns
        """
        df_processed = df.copy()
        
        for column in columns:
            if column not in df_processed.columns:
                logger.warning(f"Column '{column}' not found, skipping")
                continue
            
            if df_processed[column].dtype not in ['int64', 'float64', 'int32', 'float32']:
                logger.warning(f"Column '{column}' is not numeric, skipping")
                continue
            
            if method == 'standard':
                scaler = StandardScaler()
                df_processed[column] = scaler.fit_transform(df_processed[[column]]).ravel()
                self.scalers[column] = scaler
            
            elif method == 'minmax':
                scaler = MinMaxScaler()
                df_processed[column] = scaler.fit_transform(df_processed[[column]]).ravel()
                self.scalers[column] = scaler
            
            logger.info(f"Normalized column '{column}' using {method} scaling")
        
        return df_processed
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str], method: str = 'label') -> pd.DataFrame:
        """
        Encode categorical columns.
        
        Args:
            df: Input DataFrame
            columns: Categorical columns to encode
            method: Encoding method ('label', 'onehot')
        
        Returns:
            DataFrame with encoded columns
        """
        df_processed = df.copy()
        
        for column in columns:
            if column not in df_processed.columns:
                logger.warning(f"Column '{column}' not found, skipping")
                continue
            
            if method == 'label':
                encoder = LabelEncoder()
                df_processed[column] = encoder.fit_transform(df_processed[column].astype(str))
                self.encoders[column] = encoder
                logger.info(f"Label encoded column '{column}'")
            
            elif method == 'onehot':
                dummies = pd.get_dummies(df_processed[column], prefix=column)
                df_processed = pd.concat([df_processed.drop(columns=[column]), dummies], axis=1)
                logger.info(f"One-hot encoded column '{column}' into {len(dummies.columns)} columns")
        
        return df_processed
    
    def feature_selection(self, df: pd.DataFrame, columns: List[str], action: str = 'keep') -> pd.DataFrame:
        """
        Select features by keeping or dropping specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to keep or drop
            action: 'keep' or 'drop'
        
        Returns:
            DataFrame with selected features
        """
        if action == 'keep':
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Columns not found: {missing_cols}")
            df_selected = df[[col for col in columns if col in df.columns]]
            logger.info(f"Kept {len(df_selected.columns)} columns")
        
        elif action == 'drop':
            df_selected = df.drop(columns=[col for col in columns if col in df.columns], errors='ignore')
            logger.info(f"Dropped {len(columns)} columns, remaining: {len(df_selected.columns)}")
        
        else:
            logger.warning(f"Unknown action '{action}', returning original DataFrame")
            return df
        
        return df_selected
    
    def create_datetime_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Extract datetime features from datetime columns.
        
        Args:
            df: Input DataFrame
            columns: Datetime columns to process
        
        Returns:
            DataFrame with extracted datetime features
        """
        df_processed = df.copy()
        
        for column in columns:
            if column not in df_processed.columns:
                logger.warning(f"Column '{column}' not found, skipping")
                continue
            
            try:
                df_processed[column] = pd.to_datetime(df_processed[column])
                df_processed[f'{column}_year'] = df_processed[column].dt.year
                df_processed[f'{column}_month'] = df_processed[column].dt.month
                df_processed[f'{column}_day'] = df_processed[column].dt.day
                df_processed[f'{column}_dayofweek'] = df_processed[column].dt.dayofweek
                logger.info(f"Extracted datetime features from column '{column}'")
            except Exception as e:
                logger.warning(f"Failed to extract datetime features from '{column}': {e}")
        
        return df_processed
    
    def aggregate_features(self, df: pd.DataFrame, aggregations: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Create aggregated features.
        
        Args:
            df: Input DataFrame
            aggregations: Dictionary with aggregation configuration
                        e.g., {'group_col': 'category', 'agg_col': 'value', 'function': 'mean'}
        
        Returns:
            DataFrame with aggregated features
        """
        df_processed = df.copy()
        
        for agg_name, agg_config in aggregations.items():
            group_col = agg_config.get('group_by')
            agg_col = agg_config.get('column')
            func = agg_config.get('function', 'mean')
            
            if not group_col or not agg_col:
                logger.warning(f"Incomplete aggregation config for '{agg_name}', skipping")
                continue
            
            if group_col not in df_processed.columns or agg_col not in df_processed.columns:
                logger.warning(f"Columns for aggregation '{agg_name}' not found, skipping")
                continue
            
            try:
                aggregated = df_processed.groupby(group_col)[agg_col].agg(func).reset_index()
                aggregated.columns = [group_col, f'{agg_name}_{func}']
                df_processed = df_processed.merge(aggregated, on=group_col, how='left')
                logger.info(f"Created aggregated feature '{agg_name}_{func}'")
            except Exception as e:
                logger.warning(f"Failed to create aggregation '{agg_name}': {e}")
        
        return df_processed
    
    def preprocess(self, df: pd.DataFrame, preprocessing_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply all preprocessing operations based on configuration.
        
        Args:
            df: Input DataFrame
            preprocessing_config: Configuration dictionary with preprocessing steps
        
        Returns:
            Preprocessed DataFrame
        """
        df_processed = df.copy()
        logger.info("Starting data preprocessing process")
        
        # Feature selection
        if preprocessing_config.get('feature_selection', {}).get('enabled', False):
            selection_config = preprocessing_config['feature_selection']
            df_processed = self.feature_selection(
                df_processed,
                columns=selection_config.get('columns', []),
                action=selection_config.get('action', 'keep')
            )
        
        # Normalization
        if preprocessing_config.get('normalize', {}).get('enabled', False):
            normalize_config = preprocessing_config['normalize']
            df_processed = self.normalize(
                df_processed,
                columns=normalize_config.get('columns', []),
                method=normalize_config.get('method', 'standard')
            )
        
        # Categorical encoding
        if preprocessing_config.get('encode_categorical', {}).get('enabled', False):
            encode_config = preprocessing_config['encode_categorical']
            df_processed = self.encode_categorical(
                df_processed,
                columns=encode_config.get('columns', []),
                method=encode_config.get('method', 'label')
            )
        
        # Datetime features
        if preprocessing_config.get('datetime_features', {}).get('enabled', False):
            datetime_config = preprocessing_config['datetime_features']
            df_processed = self.create_datetime_features(
                df_processed,
                columns=datetime_config.get('columns', [])
            )
        
        # Aggregated features
        if preprocessing_config.get('aggregate_features', {}).get('enabled', False):
            aggregate_config = preprocessing_config['aggregate_features']
            df_processed = self.aggregate_features(
                df_processed,
                aggregations=aggregate_config.get('aggregations', {})
            )
        
        logger.info("Data preprocessing process completed")
        return df_processed
