from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Union
import hashlib
import json
import warnings
from datetime import datetime
import os

# ML & Stats imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression

# Statistical tests
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, ks_2samp
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils

# Optional advanced libraries
try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from kmodes.kprototypes import KPrototypes

    KPROTOTYPES_AVAILABLE = True
except ImportError:
    KPROTOTYPES_AVAILABLE = False

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)


# ============= Data Structures =============
@dataclass
class DataInfo:
    numeric_cols: List[str]
    categorical_cols: List[str]
    n_rows: int
    n_cols: int
    missing_info: Dict[str, int]
    data_types: Dict[str, str]
    basic_stats: Dict[str, Any]


@dataclass
class ClusterResult:
    labels: np.ndarray
    metrics: Dict[str, float]
    profiles: Dict[int, Dict]
    feature_names: List[str]
    projection_2d: np.ndarray
    projection_3d: Optional[np.ndarray]
    stability_score: Optional[float]
    algorithm_info: Dict[str, Any]


# ============= Utility Functions =============
def _hash_dict(d: dict) -> str:
    """Create hash from dictionary for caching"""
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()


def _safe_list(x) -> list:
    """Safely convert to list"""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (np.ndarray, pd.Series)):
        return x.tolist()
    return [x]


def _safe_float(x, default=0.0) -> float:
    """Safely convert to float"""
    try:
        return float(x)
    except:
        return default


# ============= Data Service =============
class DataService:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.info = None
        self._cache = {}

    def load(self) -> DataInfo:
        """Load and analyze data"""
        try:
            self.df = pd.read_csv(self.csv_path)

            # Identify column types
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

            # Handle datetime columns
            for col in categorical_cols[:]:
                try:
                    pd.to_datetime(self.df[col])
                    categorical_cols.remove(col)
                except:
                    pass

            # Missing value analysis
            missing_info = self.df.isnull().sum().to_dict()

            # Basic imputation
            for col in numeric_cols:
                if self.df[col].isnull().any():
                    self.df[col].fillna(self.df[col].median(), inplace=True)

            for col in categorical_cols:
                if self.df[col].isnull().any():
                    mode_val = self.df[col].mode()
                    if len(mode_val) > 0:
                        self.df[col].fillna(mode_val[0], inplace=True)
                    else:
                        self.df[col].fillna('Missing', inplace=True)

            # Compute basic statistics
            basic_stats = {
                'numeric_summary': {},
                'categorical_summary': {}
            }

            for col in numeric_cols:
                basic_stats['numeric_summary'][col] = {
                    'mean': float(self.df[col].mean()),
                    'std': float(self.df[col].std()),
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max()),
                    'q25': float(self.df[col].quantile(0.25)),
                    'q50': float(self.df[col].quantile(0.50)),
                    'q75': float(self.df[col].quantile(0.75)),
                    'skew': float(self.df[col].skew()),
                    'kurtosis': float(self.df[col].kurtosis())
                }

            for col in categorical_cols:
                value_counts = self.df[col].value_counts()
                basic_stats['categorical_summary'][col] = {
                    'unique_values': int(self.df[col].nunique()),
                    'top_values': value_counts.head(10).to_dict(),
                    'entropy': float(stats.entropy(value_counts.values))
                }

            self.info = DataInfo(
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                n_rows=len(self.df),
                n_cols=len(self.df.columns),
                missing_info=missing_info,
                data_types={col: str(self.df[col].dtype) for col in self.df.columns},
                basic_stats=basic_stats
            )

            return self.info

        except Exception as e:
            print(f"Error loading data: {e}")
            raise


# ============= Advanced Clustering Engine =============
class ClusterEngine:
    def __init__(self, df: pd.DataFrame, info: DataInfo):
        self.df = df
        self.info = info
        self._cache = {}
        self.last_result = None

    def _make_matrix_for_vector_algs(
            self,
            features_num: List[str],
            features_cat: List[str],
            scale_numeric: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare data matrix for vector-based algorithms"""

        transformers = []

        # Numeric features
        if features_num:
            if scale_numeric:
                transformers.append(
                    ('num', StandardScaler(), features_num)
                )
            else:
                transformers.append(
                    ('num', 'passthrough', features_num)
                )

        # Categorical features
        if features_cat:
            transformers.append(
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), features_cat)
            )

        if not transformers:
            raise ValueError("No features selected")

        preprocessor = ColumnTransformer(transformers)
        X = preprocessor.fit_transform(self.df)

        # Get feature names
        feature_names = []
        if features_num:
            feature_names.extend(features_num)
        if features_cat:
            cat_encoder = preprocessor.named_transformers_.get('cat')
            if cat_encoder:
                for i, col in enumerate(features_cat):
                    categories = cat_encoder.categories_[i][1:]  # Skip first (dropped)
                    feature_names.extend([f"{col}_{cat}" for cat in categories])

        return X, feature_names

    def _separate_for_kprotos(
            self,
            features_num: List[str],
            features_cat: List[str]
    ) -> Tuple[np.ndarray, List[int]]:
        """Prepare data for k-prototypes"""

        X_num = self.df[features_num].values if features_num else np.empty((len(self.df), 0))
        X_cat = self.df[features_cat].values if features_cat else np.empty((len(self.df), 0))

        if X_cat.shape[1] > 0:
            # Convert to string type
            X_cat = X_cat.astype(str)

        X = np.hstack([X_num, X_cat])
        cat_indices = list(range(len(features_num), len(features_num) + len(features_cat)))

        return X, cat_indices

    def _internal_scores(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate internal clustering metrics"""

        metrics = {}

        # Check validity
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])

        if n_clusters < 2 or n_clusters >= len(X):
            return metrics

        try:
            # Silhouette Score (-1 to 1, higher is better)
            metrics['silhouette'] = float(silhouette_score(X, labels))
        except:
            pass

        try:
            # Davies-Bouldin Index (lower is better)
            metrics['davies_bouldin'] = float(davies_bouldin_score(X, labels))
        except:
            pass

        try:
            # Calinski-Harabasz Index (higher is better)
            metrics['calinski_harabasz'] = float(calinski_harabasz_score(X, labels))
        except:
            pass

        # Cluster sizes
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        metrics['n_clusters'] = int(n_clusters)
        metrics['cluster_sizes'] = {int(k): int(v) for k, v in zip(unique, counts)}

        # Noise points (for HDBSCAN)
        if -1 in labels:
            metrics['n_noise'] = int(np.sum(labels == -1))
            metrics['noise_ratio'] = float(np.mean(labels == -1))

        return metrics

    def _cluster_profiles(
            self,
            labels: np.ndarray,
            features_num: List[str],
            features_cat: List[str]
    ) -> Dict[int, Dict]:
        """Create detailed profiles for each cluster"""

        profiles = {}
        df_labeled = self.df.copy()
        df_labeled['_cluster'] = labels

        # Global stats for comparison
        global_stats = {}
        for col in features_num:
            global_stats[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std()
            }

        for col in features_cat:
            global_stats[col] = self.df[col].value_counts(normalize=True).to_dict()

        # Profile each cluster
        for cluster_id in np.unique(labels):
            if cluster_id < 0:  # Skip noise
                continue

            cluster_data = df_labeled[df_labeled['_cluster'] == cluster_id]
            n_cluster = len(cluster_data)

            profile = {
                'size': int(n_cluster),
                'share': float(n_cluster / len(df_labeled)),
                'numeric_summary': {},
                'categorical_summary': {}
            }

            # Numeric features
            for col in features_num:
                col_data = cluster_data[col]
                profile['numeric_summary'][col] = {
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'q25': float(col_data.quantile(0.25)),
                    'q75': float(col_data.quantile(0.75)),
                    'delta_from_global': float(col_data.mean() - global_stats[col]['mean']),
                    'effect_size': float((col_data.mean() - global_stats[col]['mean']) / global_stats[col]['std']) if
                    global_stats[col]['std'] > 0 else 0
                }

            # Categorical features
            for col in features_cat:
                value_counts = cluster_data[col].value_counts(normalize=True)
                top_values = value_counts.head(5).to_dict()

                # Calculate lift
                lift_values = {}
                for val, prop in top_values.items():
                    global_prop = global_stats[col].get(val, 0.001)
                    lift_values[val] = float(prop / global_prop)

                profile['categorical_summary'][col] = {
                    'top_values': top_values,
                    'lift': lift_values,
                    'mode': cluster_data[col].mode()[0] if len(cluster_data[col].mode()) > 0 else None
                }

            profiles[int(cluster_id)] = profile

        return profiles

    def run(
            self,
            algorithm: str,
            features_num: List[str],
            features_cat: List[str],
            params: Dict[str, Any],
            scale_numeric: bool = True,
            random_state: Optional[int] = 42
    ) -> ClusterResult:
        """Run clustering algorithm"""

        # Create cache key
        cache_key = _hash_dict({
            'algorithm': algorithm,
            'features_num': sorted(features_num),
            'features_cat': sorted(features_cat),
            'params': params,
            'scale_numeric': scale_numeric,
            'random_state': random_state
        })

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Prepare data
        if algorithm == 'kprototypes' and KPROTOTYPES_AVAILABLE:
            X, cat_indices = self._separate_for_kprotos(features_num, features_cat)
            feature_names = features_num + features_cat
        else:
            X, feature_names = self._make_matrix_for_vector_algs(features_num, features_cat, scale_numeric)

        # Run algorithm
        if algorithm == 'kmeans':
            model = KMeans(
                n_clusters=params.get('n_clusters', 3),
                init=params.get('init', 'k-means++'),
                n_init=params.get('n_init', 10),
                max_iter=params.get('max_iter', 300),
                random_state=random_state
            )
            labels = model.fit_predict(X)

        elif algorithm == 'agglomerative':
            model = AgglomerativeClustering(
                n_clusters=params.get('n_clusters', 3),
                linkage=params.get('linkage', 'ward'),
                affinity=params.get('affinity', 'euclidean')
            )
            labels = model.fit_predict(X)

        elif algorithm == 'gmm':
            model = GaussianMixture(
                n_components=params.get('n_clusters', 3),
                covariance_type=params.get('covariance_type', 'full'),
                n_init=params.get('n_init', 1),
                random_state=random_state
            )
            labels = model.fit_predict(X)

        elif algorithm == 'hdbscan' and HDBSCAN_AVAILABLE:
            model = hdbscan.HDBSCAN(
                min_cluster_size=params.get('min_cluster_size', 5),
                min_samples=params.get('min_samples', None),
                epsilon=params.get('epsilon', 0.0),
                cluster_selection_method=params.get('cluster_selection_method', 'eom')
            )
            labels = model.fit_predict(X)

        elif algorithm == 'kprototypes' and KPROTOTYPES_AVAILABLE:
            model = KPrototypes(
                n_clusters=params.get('n_clusters', 3),
                gamma=params.get('gamma', None),
                n_init=params.get('n_init', 1),
                random_state=random_state
            )
            labels = model.fit_predict(X, categorical=cat_indices)

        else:
            raise ValueError(f"Algorithm {algorithm} not available")

        # Calculate metrics
        metrics = self._internal_scores(X, labels)

        # Create profiles
        profiles = self._cluster_profiles(labels, features_num, features_cat)

        # 2D and 3D projections
        projection_2d = self._create_projection(X, method='pca', n_components=2)
        projection_3d = self._create_projection(X, method='pca', n_components=3)

        # Algorithm info
        algorithm_info = {
            'algorithm': algorithm,
            'params': params,
            'n_features': len(feature_names),
            'n_samples': len(X)
        }

        result = ClusterResult(
            labels=labels,
            metrics=metrics,
            profiles=profiles,
            feature_names=feature_names,
            projection_2d=projection_2d,
            projection_3d=projection_3d,
            stability_score=None,
            algorithm_info=algorithm_info
        )

        self._cache[cache_key] = result
        self.last_result = result

        return result

    def sweep_k(
            self,
            algorithm: str,
            features_num: List[str],
            features_cat: List[str],
            k_range: range,
            scale_numeric: bool = True,
            random_state: Optional[int] = 42
    ) -> Dict[str, Any]:
        """Sweep over different k values"""

        results = []

        for k in k_range:
            params = {'n_clusters': k} if algorithm != 'hdbscan' else {'min_cluster_size': k}

            try:
                result = self.run(
                    algorithm=algorithm,
                    features_num=features_num,
                    features_cat=features_cat,
                    params=params,
                    scale_numeric=scale_numeric,
                    random_state=random_state
                )

                record = {
                    'k': k,
                    'silhouette': result.metrics.get('silhouette', np.nan),
                    'davies_bouldin': result.metrics.get('davies_bouldin', np.nan),
                    'calinski_harabasz': result.metrics.get('calinski_harabasz', np.nan),
                    'n_clusters': result.metrics.get('n_clusters', 0)
                }
                results.append(record)

            except Exception as e:
                print(f"Error at k={k}: {e}")
                continue

        # Find best k
        best_k = None
        if results:
            # Priority: silhouette > calinski_harabasz > davies_bouldin (inverted)
            valid_results = [r for r in results if not np.isnan(r['silhouette'])]

            if valid_results:
                best_k = max(valid_results, key=lambda x: x['silhouette'])['k']
            elif results:
                valid_results = [r for r in results if not np.isnan(r['calinski_harabasz'])]
                if valid_results:
                    best_k = max(valid_results, key=lambda x: x['calinski_harabasz'])['k']

        return {
            'records': results,
            'best_k': best_k,
            'algorithm': algorithm
        }

    def stability_analysis(
            self,
            algorithm: str,
            features_num: List[str],
            features_cat: List[str],
            params: Dict[str, Any],
            n_runs: int = 10,
            subsample_ratio: float = 0.8,
            random_state: Optional[int] = 42
    ) -> Dict[str, Any]:
        """Analyze clustering stability"""

        np.random.seed(random_state)

        # Prepare data once
        if algorithm == 'kprototypes' and KPROTOTYPES_AVAILABLE:
            X, cat_indices = self._separate_for_kprotos(features_num, features_cat)
        else:
            X, feature_names = self._make_matrix_for_vector_algs(features_num, features_cat)

        # Run multiple times
        all_labels = []
        n_samples = len(X)
        subsample_size = int(n_samples * subsample_ratio)

        for i in range(n_runs):
            # Random subsample
            indices = np.random.choice(n_samples, subsample_size, replace=False)
            X_sub = X[indices]

            # Run clustering
            if algorithm == 'kmeans':
                model = KMeans(
                    n_clusters=params.get('n_clusters', 3),
                    random_state=random_state + i if random_state else None
                )
                labels_sub = model.fit_predict(X_sub)

            elif algorithm == 'agglomerative':
                model = AgglomerativeClustering(
                    n_clusters=params.get('n_clusters', 3)
                )
                labels_sub = model.fit_predict(X_sub)

            # Store with original indices
            labels_full = np.full(n_samples, -1)
            labels_full[indices] = labels_sub
            all_labels.append(labels_full)

        # Calculate pairwise ARI scores
        ari_scores = []
        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                # Get common indices
                mask = (all_labels[i] >= 0) & (all_labels[j] >= 0)
                if np.sum(mask) > 0:
                    ari = adjusted_rand_score(all_labels[i][mask], all_labels[j][mask])
                    ari_scores.append(ari)

        return {
            'mean_ari': float(np.mean(ari_scores)) if ari_scores else 0.0,
            'std_ari': float(np.std(ari_scores)) if ari_scores else 0.0,
            'n_pairs': len(ari_scores),
            'ari_scores': ari_scores
        }

    def _create_projection(
            self,
            X: np.ndarray,
            method: str = 'pca',
            n_components: int = 2
    ) -> np.ndarray:
        """Create dimensionality reduction projection"""

        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'tsne' and n_components == 2:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) - 1))
        elif method == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        else:
            reducer = PCA(n_components=n_components, random_state=42)

        try:
            projection = reducer.fit_transform(X)
        except:
            # Fallback to PCA
            reducer = PCA(n_components=n_components, random_state=42)
            projection = reducer.fit_transform(X)

        return projection


# ============= Statistical Analysis Engine =============
class StatisticalAnalyzer:
    def __init__(self, df: pd.DataFrame, info: DataInfo):
        self.df = df
        self.info = info

    def run_tests(self, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive statistical tests"""

        results = {
            'normality_tests': {},
            'correlation_tests': {},
            'hypothesis_tests': {},
            'distribution_tests': {}
        }

        # Normality tests for numeric columns
        for col in self.info.numeric_cols[:10]:  # Limit to first 10
            try:
                statistic, p_value = stats.shapiro(self.df[col].dropna())
                results['normality_tests'][col] = {
                    'test': 'Shapiro-Wilk',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05
                }
            except:
                pass

        # Correlation analysis
        if len(self.info.numeric_cols) > 1:
            corr_matrix = self.df[self.info.numeric_cols].corr()

            # Find strongest correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:  # Strong correlation
                        corr_pairs.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': float(corr_val),
                            'strength': 'strong' if abs(corr_val) > 0.7 else 'moderate'
                        })

            results['correlation_tests']['strong_correlations'] = corr_pairs
            results['correlation_tests']['correlation_matrix'] = corr_matrix.to_dict()

        # If target column specified, run specific tests
        if target_col and target_col in self.df.columns:

            # Check if target is categorical or numeric
            if target_col in self.info.categorical_cols:
                # Chi-square tests with other categorical variables
                for col in self.info.categorical_cols:
                    if col != target_col:
                        try:
                            contingency = pd.crosstab(self.df[col], self.df[target_col])
                            chi2, p_value, dof, expected = chi2_contingency(contingency)

                            results['hypothesis_tests'][f'{col}_vs_{target_col}'] = {
                                'test': 'Chi-square',
                                'statistic': float(chi2),
                                'p_value': float(p_value),
                                'degrees_of_freedom': int(dof),
                                'significant': p_value < 0.05
                            }
                        except:
                            pass

                # ANOVA for numeric variables
                for col in self.info.numeric_cols[:10]:
                    try:
                        groups = [group[col].dropna() for name, group in self.df.groupby(target_col)]
                        if len(groups) >= 2:
                            f_stat, p_value = f_oneway(*groups)

                            results['hypothesis_tests'][f'{col}_by_{target_col}'] = {
                                'test': 'ANOVA',
                                'statistic': float(f_stat),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05
                            }
                    except:
                        pass

            elif target_col in self.info.numeric_cols:
                # T-tests for binary categorical variables
                for col in self.info.categorical_cols:
                    if self.df[col].nunique() == 2:
                        try:
                            groups = self.df.groupby(col)[target_col].apply(list).values
                            t_stat, p_value = ttest_ind(groups[0], groups[1])

                            results['hypothesis_tests'][f'{target_col}_by_{col}'] = {
                                'test': 't-test',
                                'statistic': float(t_stat),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05
                            }
                        except:
                            pass

        return results

    def outlier_detection(self) -> Dict[str, Any]:
        """Detect outliers using multiple methods"""

        results = {
            'isolation_forest': {},
            'iqr_method': {},
            'z_score': {}
        }

        numeric_data = self.df[self.info.numeric_cols]

        # Isolation Forest
        try:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers_iso = iso_forest.fit_predict(numeric_data)

            results['isolation_forest'] = {
                'n_outliers': int(np.sum(outliers_iso == -1)),
                'outlier_ratio': float(np.mean(outliers_iso == -1)),
                'outlier_indices': np.where(outliers_iso == -1)[0].tolist()[:100]  # First 100
            }
        except:
            pass

        # IQR Method
        for col in self.info.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)

            results['iqr_method'][col] = {
                'n_outliers': int(outliers.sum()),
                'outlier_ratio': float(outliers.mean()),
                'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
            }

        # Z-score method
        for col in self.info.numeric_cols:
            z_scores = np.abs(stats.zscore(self.df[col].dropna()))
            threshold = 3
            outliers = z_scores > threshold

            results['z_score'][col] = {
                'n_outliers': int(outliers.sum()),
                'outlier_ratio': float(outliers.mean()),
                'threshold': threshold
            }

        return results


# ============= Visualization Service =============
class VisualizationService:
    def __init__(self, df: pd.DataFrame, info: DataInfo):
        self.df = df
        self.info = info

    def create_cluster_visualization(self, result: ClusterResult) -> str:
        """Create interactive cluster visualization"""

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('2D Projection', '3D Projection', 'Cluster Sizes', 'Metrics'),
            specs=[[{"type": "scatter"}, {"type": "scatter3d"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        # 2D scatter
        fig.add_trace(
            go.Scatter(
                x=result.projection_2d[:, 0],
                y=result.projection_2d[:, 1],
                mode='markers',
                marker=dict(
                    color=result.labels,
                    colorscale='Viridis',
                    size=5,
                    opacity=0.7
                ),
                text=[f'Cluster {l}' for l in result.labels],
                name='2D Projection'
            ),
            row=1, col=1
        )

        # 3D scatter
        if result.projection_3d is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=result.projection_3d[:, 0],
                    y=result.projection_3d[:, 1],
                    z=result.projection_3d[:, 2],
                    mode='markers',
                    marker=dict(
                        color=result.labels,
                        colorscale='Viridis',
                        size=3,
                        opacity=0.7
                    ),
                    text=[f'Cluster {l}' for l in result.labels],
                    name='3D Projection'
                ),
                row=1, col=2
            )

        # Cluster sizes
        cluster_sizes = result.metrics.get('cluster_sizes', {})
        fig.add_trace(
            go.Bar(
                x=list(cluster_sizes.keys()),
                y=list(cluster_sizes.values()),
                marker=dict(color=list(cluster_sizes.keys()), colorscale='Viridis'),
                name='Cluster Sizes'
            ),
            row=2, col=1
        )

        # Metrics bar chart
        metrics_to_plot = {
            'Silhouette': result.metrics.get('silhouette', 0),
            'Davies-Bouldin': -result.metrics.get('davies_bouldin', 0),  # Invert for visualization
            'Calinski-Harabasz (scaled)': result.metrics.get('calinski_harabasz', 0) / 100
        }

        fig.add_trace(
            go.Bar(
                x=list(metrics_to_plot.keys()),
                y=list(metrics_to_plot.values()),
                marker=dict(color=['green', 'orange', 'blue']),
                name='Metrics'
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            title_text=f"Clustering Results - {result.algorithm_info['algorithm'].upper()}",
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)'
        )

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def create_elbow_plot(self, sweep_results: Dict) -> str:
        """Create elbow plot for k selection"""

        records = sweep_results['records']
        if not records:
            return json.dumps({})

        df_records = pd.DataFrame(records)

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index')
        )

        # Silhouette
        fig.add_trace(
            go.Scatter(
                x=df_records['k'],
                y=df_records['silhouette'],
                mode='lines+markers',
                marker=dict(size=8, color='blue'),
                line=dict(color='blue', width=2),
                name='Silhouette'
            ),
            row=1, col=1
        )

        # Davies-Bouldin
        fig.add_trace(
            go.Scatter(
                x=df_records['k'],
                y=df_records['davies_bouldin'],
                mode='lines+markers',
                marker=dict(size=8, color='red'),
                line=dict(color='red', width=2),
                name='Davies-Bouldin'
            ),
            row=1, col=2
        )

        # Calinski-Harabasz
        fig.add_trace(
            go.Scatter(
                x=df_records['k'],
                y=df_records['calinski_harabasz'],
                mode='lines+markers',
                marker=dict(size=8, color='green'),
                line=dict(color='green', width=2),
                name='Calinski-Harabasz'
            ),
            row=1, col=3
        )

        # Mark best k
        if sweep_results.get('best_k'):
            best_k = sweep_results['best_k']
            best_record = next(r for r in records if r['k'] == best_k)

            # Add vertical line at best k
            for col in range(1, 4):
                fig.add_vline(
                    x=best_k,
                    line_dash="dash",
                    line_color="gray",
                    row=1, col=col
                )

        fig.update_layout(
            height=400,
            title_text=f"Elbow Analysis - {sweep_results['algorithm'].upper()}",
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)'
        )

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def create_statistical_plots(self, stats_results: Dict) -> Dict[str, str]:
        """Create statistical analysis plots"""

        plots = {}

        # Correlation heatmap
        if 'correlation_matrix' in stats_results.get('correlation_tests', {}):
            corr_matrix = pd.DataFrame(stats_results['correlation_tests']['correlation_matrix'])

            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                title="Correlation Matrix",
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )

            plots['correlation_heatmap'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # P-value distribution
        if 'hypothesis_tests' in stats_results:
            p_values = [test['p_value'] for test in stats_results['hypothesis_tests'].values()]

            if p_values:
                fig = go.Figure(data=[
                    go.Histogram(
                        x=p_values,
                        nbinsx=20,
                        marker_color='blue',
                        opacity=0.7
                    )
                ])

                fig.add_vline(x=0.05, line_dash="dash", line_color="red",
                              annotation_text="α = 0.05")

                fig.update_layout(
                    title="P-value Distribution",
                    xaxis_title="P-value",
                    yaxis_title="Frequency",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0.1)'
                )

                plots['p_value_dist'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return plots


# ============= Initialize Services =============
DATA_PATH = os.environ.get('DATA_PATH', 'data.csv')
data_service = None
cluster_engine = None
stat_analyzer = None
viz_service = None


def initialize_services():
    """Initialize all services"""
    global data_service, cluster_engine, stat_analyzer, viz_service

    try:
        data_service = DataService(DATA_PATH)
        info = data_service.load()

        cluster_engine = ClusterEngine(data_service.df, info)
        stat_analyzer = StatisticalAnalyzer(data_service.df, info)
        viz_service = VisualizationService(data_service.df, info)

        print(f"✅ Services initialized successfully")
        print(f"📊 Data: {info.n_rows} rows, {info.n_cols} columns")
        print(f"🔢 Numeric columns: {len(info.numeric_cols)}")
        print(f"📝 Categorical columns: {len(info.categorical_cols)}")

        return True
    except Exception as e:
        print(f"❌ Error initializing services: {e}")
        return False


# ============= Flask Routes =============

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'services': {
            'data_loaded': data_service is not None,
            'cluster_engine': cluster_engine is not None,
            'stat_analyzer': stat_analyzer is not None,
            'viz_service': viz_service is not None
        },
        'optional_algorithms': {
            'hdbscan': HDBSCAN_AVAILABLE,
            'umap': UMAP_AVAILABLE,
            'kprototypes': KPROTOTYPES_AVAILABLE
        }
    })


@app.route('/api/schema')
def get_schema():
    """Get data schema and info"""
    if not data_service:
        return jsonify({'error': 'Services not initialized'}), 500

    return jsonify({
        'n_rows': data_service.info.n_rows,
        'n_cols': data_service.info.n_cols,
        'numeric_cols': data_service.info.numeric_cols,
        'categorical_cols': data_service.info.categorical_cols,
        'missing_info': data_service.info.missing_info,
        'data_types': data_service.info.data_types,
        'basic_stats': data_service.info.basic_stats,
        'optional_algorithms': {
            'hdbscan': HDBSCAN_AVAILABLE,
            'umap': UMAP_AVAILABLE,
            'kprototypes': KPROTOTYPES_AVAILABLE
        }
    })


@app.route('/api/cluster', methods=['POST'])
def run_clustering():
    """Run clustering algorithm"""
    if not cluster_engine:
        return jsonify({'error': 'Cluster engine not initialized'}), 500

    try:
        data = request.json

        result = cluster_engine.run(
            algorithm=data.get('algorithm', 'kmeans'),
            features_num=_safe_list(data.get('features_numeric', [])),
            features_cat=_safe_list(data.get('features_categorical', [])),
            params=data.get('params', {}),
            scale_numeric=data.get('scale_numeric', True),
            random_state=data.get('random_state', 42)
        )

        # Create visualization
        viz_json = viz_service.create_cluster_visualization(result)

        return jsonify({
            'labels': result.labels.tolist(),
            'metrics': result.metrics,
            'profiles': result.profiles,
            'feature_names': result.feature_names,
            'projection_2d': result.projection_2d.tolist(),
            'projection_3d': result.projection_3d.tolist() if result.projection_3d is not None else None,
            'algorithm_info': result.algorithm_info,
            'visualization': viz_json
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/cluster/sweep', methods=['POST'])
def sweep_clustering():
    """Sweep k values for clustering"""
    if not cluster_engine:
        return jsonify({'error': 'Cluster engine not initialized'}), 500

    try:
        data = request.json

        k_min = data.get('k_min', 2)
        k_max = data.get('k_max', 10)

        results = cluster_engine.sweep_k(
            algorithm=data.get('algorithm', 'kmeans'),
            features_num=_safe_list(data.get('features_numeric', [])),
            features_cat=_safe_list(data.get('features_categorical', [])),
            k_range=range(k_min, k_max + 1),
            scale_numeric=data.get('scale_numeric', True),
            random_state=data.get('random_state', 42)
        )

        # Create elbow plot
        elbow_plot = viz_service.create_elbow_plot(results)

        return jsonify({
            'records': results['records'],
            'best_k': results['best_k'],
            'algorithm': results['algorithm'],
            'visualization': elbow_plot
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/cluster/stability', methods=['POST'])
def cluster_stability():
    """Analyze clustering stability"""
    if not cluster_engine:
        return jsonify({'error': 'Cluster engine not initialized'}), 500

    try:
        data = request.json

        results = cluster_engine.stability_analysis(
            algorithm=data.get('algorithm', 'kmeans'),
            features_num=_safe_list(data.get('features_numeric', [])),
            features_cat=_safe_list(data.get('features_categorical', [])),
            params=data.get('params', {}),
            n_runs=data.get('n_runs', 10),
            subsample_ratio=data.get('subsample_ratio', 0.8),
            random_state=data.get('random_state', 42)
        )

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/statistics', methods=['POST'])
def run_statistics():
    """Run statistical analysis"""
    if not stat_analyzer:
        return jsonify({'error': 'Statistical analyzer not initialized'}), 500

    try:
        data = request.json
        target_col = data.get('target_column')

        # Run tests
        test_results = stat_analyzer.run_tests(target_col)

        # Run outlier detection
        outlier_results = stat_analyzer.outlier_detection()

        # Create visualizations
        stat_plots = viz_service.create_statistical_plots(test_results)

        return jsonify({
            'test_results': test_results,
            'outlier_results': outlier_results,
            'visualizations': stat_plots
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/export', methods=['POST'])
def export_results():
    """Export analysis results"""
    if not cluster_engine or not cluster_engine.last_result:
        return jsonify({'error': 'No clustering results available'}), 400

    try:
        result = cluster_engine.last_result

        # Prepare export data
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'algorithm': result.algorithm_info,
            'metrics': result.metrics,
            'profiles': result.profiles,
            'n_clusters': result.metrics.get('n_clusters', 0),
            'cluster_sizes': result.metrics.get('cluster_sizes', {}),
            'feature_names': result.feature_names
        }

        # Add cluster assignments to dataframe
        df_export = data_service.df.copy()
        df_export['cluster'] = result.labels

        # Convert to CSV
        csv_data = df_export.to_csv(index=False)

        return jsonify({
            'analysis_summary': export_data,
            'csv_data': csv_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ============= Main =============
if __name__ == '__main__':
    if initialize_services():
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to initialize services. Please check your data file.")