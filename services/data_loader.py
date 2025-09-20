"""
Data Loader Service
Handles data loading, preprocessing, and filtering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import pickle
import os
from datetime import datetime
from config import Config

class DataLoader:
    """Data loading and preprocessing service"""

    def __init__(self, data_path: str = None):
        """Initialize data loader"""
        self.data_path = data_path or Config.DATA_PATH
        self.df_raw = None
        self.df_processed = None
        self.encoders = {}
        self.scalers = {}

    def load_and_preprocess(self) -> pd.DataFrame:
        """Load and preprocess data"""
        print(f"Loading data from {self.data_path}...")

        # Load raw data
        self.df_raw = pd.read_csv(self.data_path)

        # Create a copy for processing
        df = self.df_raw.copy()

        # Basic preprocessing
        df = self._clean_data(df)
        df = self._handle_missing_values(df)
        df = self._convert_types(df)
        df = self._feature_engineering(df)

        self.df_processed = df

        # Save preprocessed data to cache
        self._save_to_cache(df)

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data"""
        # Remove duplicates
        df = df.drop_duplicates()

        # Trim string columns
        str_columns = df.select_dtypes(include=['object']).columns
        for col in str_columns:
            df[col] = df[col].str.strip()

        # Standardize column names
        df.columns = df.columns.str.replace(' ', '_')

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        # Check missing values
        missing_info = df.isnull().sum()

        if missing_info.sum() > 0:
            print(f"Missing values found: \n{missing_info[missing_info > 0]}")

            # Strategy for different columns
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    # Numerical: fill with median
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    # Categorical: fill with mode or 'Unknown'
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col].fillna(mode_val[0], inplace=True)
                    else:
                        df[col].fillna('Unknown', inplace=True)

        return df

    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types"""
        # Convert Survey_Date to datetime
        if 'Survey_Date' in df.columns:
            df['Survey_Date'] = pd.to_datetime(df['Survey_Date'], errors='coerce')

        # Ensure numerical columns are numeric
        for col in Config.NUMERICAL_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Ensure categorical columns are string
        for col in Config.CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features"""

        # Age groups
        if 'Age' in df.columns:
            df['Age_Group'] = pd.cut(
                df['Age'],
                bins=[0, 25, 35, 45, 55, 100],
                labels=['<25', '25-34', '35-44', '45-54', '55+']
            )

        # Work intensity
        if 'Hours_Per_Week' in df.columns:
            df['Work_Intensity'] = pd.cut(
                df['Hours_Per_Week'],
                bins=[0, 35, 40, 50, 100],
                labels=['Part-time', 'Standard', 'Overtime', 'Excessive']
            )

        # Burnout risk score (composite)
        burnout_factors = []
        if 'Hours_Per_Week' in df.columns:
            burnout_factors.append((df['Hours_Per_Week'] - 40) / 10)
        if 'Work_Life_Balance_Score' in df.columns:
            burnout_factors.append((10 - df['Work_Life_Balance_Score']) / 10)
        if 'Social_Isolation_Score' in df.columns:
            burnout_factors.append(df['Social_Isolation_Score'] / 10)

        if burnout_factors:
            df['Burnout_Risk_Score'] = np.mean(burnout_factors, axis=0)
            df['Burnout_Risk_Score'] = df['Burnout_Risk_Score'].clip(0, 1)

        # Is remote worker
        if 'Work_Arrangement' in df.columns:
            df['Is_Remote'] = df['Work_Arrangement'].str.contains('Remote', case=False, na=False).astype(int)

        # Has health issues
        if 'Physical_Health_Issues' in df.columns:
            df['Has_Health_Issues'] = (~df['Physical_Health_Issues'].str.contains('None', case=False, na=False)).astype(int)

        # Experience level (based on age as proxy)
        if 'Age' in df.columns:
            df['Experience_Level'] = pd.cut(
                df['Age'],
                bins=[0, 28, 38, 50, 100],
                labels=['Entry', 'Mid', 'Senior', 'Executive']
            )

        # Work-Life Balance Category
        if 'Work_Life_Balance_Score' in df.columns:
            df['WLB_Category'] = pd.cut(
                df['Work_Life_Balance_Score'],
                bins=[0, 3, 6, 8, 10],
                labels=['Poor', 'Fair', 'Good', 'Excellent']
            )

        # Salary numeric (for analysis)
        if 'Salary_Range' in df.columns:
            salary_map = {
                '<50k': 40000,
                '50k-75k': 62500,
                '75k-100k': 87500,
                '100k-150k': 125000,
                '>150k': 175000
            }
            df['Salary_Numeric'] = df['Salary_Range'].map(salary_map)

        # Date features
        if 'Survey_Date' in df.columns and df['Survey_Date'].dtype == 'datetime64[ns]':
            df['Survey_Year'] = df['Survey_Date'].dt.year
            df['Survey_Month'] = df['Survey_Date'].dt.month
            df['Survey_Quarter'] = df['Survey_Date'].dt.quarter

        return df

    def apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to dataframe"""
        df_filtered = df.copy()

        for key, value in filters.items():
            if value is not None and key in df_filtered.columns:
                if isinstance(value, list):
                    # Multiple values filter
                    df_filtered = df_filtered[df_filtered[key].isin(value)]
                elif isinstance(value, dict):
                    # Range filter
                    if 'min' in value and value['min'] is not None:
                        df_filtered = df_filtered[df_filtered[key] >= value['min']]
                    if 'max' in value and value['max'] is not None:
                        df_filtered = df_filtered[df_filtered[key] <= value['max']]
                else:
                    # Single value filter
                    df_filtered = df_filtered[df_filtered[key] == value]

        return df_filtered

    def get_column_stats(self, df: pd.DataFrame, column: str) -> Dict:
        """Get statistics for a column"""
        stats = {}

        if column not in df.columns:
            return stats

        if df[column].dtype in ['float64', 'int64']:
            stats = {
                'type': 'numerical',
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'q25': df[column].quantile(0.25),
                'q75': df[column].quantile(0.75)
            }
        else:
            stats = {
                'type': 'categorical',
                'unique_values': df[column].nunique(),
                'top_values': df[column].value_counts().head(10).to_dict(),
                'mode': df[column].mode()[0] if len(df[column].mode()) > 0 else None
            }

        stats['missing'] = df[column].isnull().sum()
        stats['missing_pct'] = stats['missing'] / len(df) * 100

        return stats

    def get_data_summary(self, df: pd.DataFrame = None) -> Dict:
        """Get comprehensive data summary"""
        if df is None:
            df = self.df_processed

        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
            'numerical_columns': list(df.select_dtypes(include=['number']).columns)
        }

        # Add value counts for categorical
        for col in summary['categorical_columns'][:5]:  # Limit to first 5
            summary[f'{col}_distribution'] = df[col].value_counts().head(5).to_dict()

        return summary

    def encode_categorical(self, df: pd.DataFrame, method: str = 'label') -> pd.DataFrame:
        """Encode categorical variables"""
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        df_encoded = df.copy()

        if method == 'label':
            for col in Config.CATEGORICAL_FEATURES:
                if col in df_encoded.columns:
                    le = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = le

        elif method == 'ordinal':
            for col, mapping in Config.ORDINAL_MAPPINGS.items():
                if col in df_encoded.columns:
                    df_encoded[f'{col}_ordinal'] = df_encoded[col].map(mapping)

        elif method == 'onehot':
            for col in Config.CATEGORICAL_FEATURES:
                if col in df_encoded.columns and df_encoded[col].nunique() < 10:
                    dummies = pd.get_dummies(df_encoded[col], prefix=col)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)

        return df_encoded

    def scale_numerical(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        df_scaled = df.copy()

        scaler_map = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }

        scaler = scaler_map.get(method, StandardScaler())

        num_cols = [col for col in Config.NUMERICAL_FEATURES if col in df_scaled.columns]
        if num_cols:
            df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
            self.scalers['numerical'] = scaler

        return df_scaled

    def _save_to_cache(self, df: pd.DataFrame):
        """Save processed data to cache"""
        cache_path = os.path.join(Config.CACHE_DIR, 'processed_data.pkl')
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)

        # Save encoders and scalers
        if self.encoders:
            with open(os.path.join(Config.MODELS_DIR, 'encoders.pkl'), 'wb') as f:
                pickle.dump(self.encoders, f)

        if self.scalers:
            with open(os.path.join(Config.MODELS_DIR, 'scalers.pkl'), 'wb') as f:
                pickle.dump(self.scalers, f)

    def load_from_cache(self) -> Optional[pd.DataFrame]:
        """Load data from cache if available"""
        cache_path = os.path.join(Config.CACHE_DIR, 'processed_data.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None