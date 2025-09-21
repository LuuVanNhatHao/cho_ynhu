from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from scipy.stats import chi2_contingency, pearsonr, ttest_ind, f_oneway
from datetime import datetime
import warnings
import os
import hashlib

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables
df = None
models_cache = {}
analysis_cache = {}
data_fingerprint = None


def convert_to_serializable(obj):
    """Convert pandas/numpy objects to JSON serializable types"""
    if pd.isna(obj):
        return None
    elif isinstance(obj, (np.integer, pd.Int64Dtype)):
        return int(obj)
    elif isinstance(obj, (np.floating, pd.Float64Dtype)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.astype(object).fillna('N/A').to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.astype(object).fillna('N/A').to_dict('records')
    return obj


def generate_data_fingerprint():
    """Generate fingerprint for current dataset"""
    global df, data_fingerprint
    if df is not None:
        data_str = f"{len(df)}-{list(df.columns)}-{df.dtypes.to_dict()}"
        data_fingerprint = hashlib.md5(data_str.encode()).hexdigest()[:8]


class DataAnalyzer:
    def __init__(self, dataframe):
        self.df = dataframe
        self.processed_df = None
        self.encoders = {}
        self.scaler = StandardScaler()

    def preprocess_data(self):
        """Advanced data preprocessing"""
        self.processed_df = self.df.copy()

        # Handle missing values
        numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.processed_df.select_dtypes(include=['object']).columns

        # Fill missing values
        for col in numeric_cols:
            self.processed_df[col].fillna(self.processed_df[col].median(), inplace=True)

        for col in categorical_cols:
            if len(self.processed_df[col].mode()) > 0:
                self.processed_df[col].fillna(self.processed_df[col].mode()[0], inplace=True)
            else:
                self.processed_df[col].fillna('Unknown', inplace=True)

        # Encode categorical variables
        for col in categorical_cols:
            if col != 'Mental_Health_Status':
                le = LabelEncoder()
                self.processed_df[f'{col}_encoded'] = le.fit_transform(self.processed_df[col].astype(str))
                self.encoders[col] = le

        return self.processed_df

    def advanced_feature_analysis(self):
        """Advanced feature analysis with comprehensive insights"""
        if self.processed_df is None:
            self.preprocess_data()

        # Prepare features
        feature_cols = [col for col in self.processed_df.columns if col.endswith('_encoded')] + \
                       ['Age', 'Hours_Per_Week', 'Work_Life_Balance_Score', 'Social_Isolation_Score']
        feature_cols = [col for col in feature_cols if col in self.processed_df.columns]

        if 'Mental_Health_Status' not in self.processed_df.columns:
            return None

        # Encode target variable
        le_target = LabelEncoder()
        y = le_target.fit_transform(self.processed_df['Mental_Health_Status'].astype(str))
        X = self.processed_df[feature_cols]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Use Random Forest as the primary model for analysis
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        rf_model.fit(X_train, y_train)

        # Get feature importances
        feature_importance = {col: float(imp) for col, imp in zip(feature_cols, rf_model.feature_importances_)}

        # Data-driven insights instead of model performance
        insights = self._generate_data_insights(feature_importance)

        # Risk factor analysis
        risk_factors = self._analyze_risk_factors()

        # Protective factor analysis
        protective_factors = self._analyze_protective_factors()

        # Population segments analysis
        segments = self._analyze_population_segments()

        # Key findings and actionable insights
        key_findings = self._extract_key_findings(feature_importance)

        return {
            'feature_importance': feature_importance,
            'data_insights': insights,
            'risk_factors': risk_factors,
            'protective_factors': protective_factors,
            'population_segments': segments,
            'key_findings': key_findings,
            'target_classes': le_target.classes_.tolist(),
            'feature_names': feature_cols,
            'sample_size': len(self.processed_df),
            'analysis_date': datetime.now().isoformat()
        }

    def _generate_data_insights(self, feature_importance):
        """Generate data-driven insights from feature importance"""
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

        insights = []
        for feature, importance in top_features:
            if 'Work_Life_Balance' in feature:
                avg_wlb = self.processed_df['Work_Life_Balance_Score'].mean()
                insights.append({
                    'factor': 'Work-Life Balance',
                    'importance': float(importance * 100),
                    'description': f'Work-life balance is the strongest predictor (影響度: {importance*100:.1f}%)',
                    'current_avg': float(avg_wlb),
                    'recommendation': 'Focus on flexible work policies and time management training'
                })
            elif 'Social_Isolation' in feature:
                avg_isolation = self.processed_df['Social_Isolation_Score'].mean()
                insights.append({
                    'factor': 'Social Isolation',
                    'importance': float(importance * 100),
                    'description': f'Social isolation significantly impacts mental health (影響度: {importance*100:.1f}%)',
                    'current_avg': float(avg_isolation),
                    'recommendation': 'Implement team building and social connection programs'
                })
            elif 'Hours_Per_Week' in feature:
                avg_hours = self.processed_df['Hours_Per_Week'].mean()
                insights.append({
                    'factor': 'Working Hours',
                    'importance': float(importance * 100),
                    'description': f'Working hours heavily influence mental wellbeing (影響度: {importance*100:.1f}%)',
                    'current_avg': float(avg_hours),
                    'recommendation': 'Monitor and limit excessive working hours'
                })
            elif 'Age' in feature:
                avg_age = self.processed_df['Age'].mean()
                insights.append({
                    'factor': 'Age Demographics',
                    'importance': float(importance * 100),
                    'description': f'Age is a significant factor in mental health patterns (影響度: {importance*100:.1f}%)',
                    'current_avg': float(avg_age),
                    'recommendation': 'Develop age-specific mental health programs'
                })

        return insights

    def _analyze_risk_factors(self):
        """Analyze key risk factors for poor mental health"""
        risk_factors = []

        # High working hours risk
        if 'Hours_Per_Week' in self.processed_df.columns:
            high_hours_threshold = self.processed_df['Hours_Per_Week'].quantile(0.75)
            high_hours_df = self.processed_df[self.processed_df['Hours_Per_Week'] > high_hours_threshold]
            if len(high_hours_df) > 0:
                risk_percentage = (high_hours_df['Mental_Health_Status'].isin(['Stress Disorder', 'Anxiety']).mean() * 100)
                risk_factors.append({
                    'factor': 'Excessive Working Hours',
                    'threshold': f'>{high_hours_threshold:.0f} hours/week',
                    'affected_population': f'{len(high_hours_df)} employees ({len(high_hours_df)/len(self.processed_df)*100:.1f}%)',
                    'risk_level': float(risk_percentage),
                    'description': f'{risk_percentage:.1f}% of employees working >50h/week show stress or anxiety symptoms'
                })

        # Poor work-life balance risk
        if 'Work_Life_Balance_Score' in self.processed_df.columns:
            low_wlb_threshold = self.processed_df['Work_Life_Balance_Score'].quantile(0.25)
            low_wlb_df = self.processed_df[self.processed_df['Work_Life_Balance_Score'] <= low_wlb_threshold]
            if len(low_wlb_df) > 0:
                risk_percentage = (low_wlb_df['Mental_Health_Status'].isin(['Stress Disorder', 'Depression']).mean() * 100)
                risk_factors.append({
                    'factor': 'Poor Work-Life Balance',
                    'threshold': f'≤{low_wlb_threshold:.1f} score',
                    'affected_population': f'{len(low_wlb_df)} employees ({len(low_wlb_df)/len(self.processed_df)*100:.1f}%)',
                    'risk_level': float(risk_percentage),
                    'description': f'{risk_percentage:.1f}% of employees with poor WLB show mental health issues'
                })

        # High social isolation risk
        if 'Social_Isolation_Score' in self.processed_df.columns:
            high_isolation_threshold = self.processed_df['Social_Isolation_Score'].quantile(0.75)
            high_isolation_df = self.processed_df[self.processed_df['Social_Isolation_Score'] >= high_isolation_threshold]
            if len(high_isolation_df) > 0:
                risk_percentage = (high_isolation_df['Mental_Health_Status'].isin(['Depression', 'Anxiety']).mean() * 100)
                risk_factors.append({
                    'factor': 'High Social Isolation',
                    'threshold': f'≥{high_isolation_threshold:.1f} score',
                    'affected_population': f'{len(high_isolation_df)} employees ({len(high_isolation_df)/len(self.processed_df)*100:.1f}%)',
                    'risk_level': float(risk_percentage),
                    'description': f'{risk_percentage:.1f}% of socially isolated employees show depression or anxiety'
                })

        return risk_factors

    def _analyze_protective_factors(self):
        """Analyze factors that protect mental health"""
        protective_factors = []

        # Good work-life balance
        if 'Work_Life_Balance_Score' in self.processed_df.columns:
            high_wlb_threshold = self.processed_df['Work_Life_Balance_Score'].quantile(0.75)
            high_wlb_df = self.processed_df[self.processed_df['Work_Life_Balance_Score'] >= high_wlb_threshold]
            if len(high_wlb_df) > 0:
                healthy_percentage = (high_wlb_df['Mental_Health_Status'] == 'Healthy').mean() * 100
                protective_factors.append({
                    'factor': 'Good Work-Life Balance',
                    'threshold': f'≥{high_wlb_threshold:.1f} score',
                    'benefited_population': f'{len(high_wlb_df)} employees ({len(high_wlb_df)/len(self.processed_df)*100:.1f}%)',
                    'protection_level': float(healthy_percentage),
                    'description': f'{healthy_percentage:.1f}% of employees with good WLB maintain healthy mental status'
                })

        # Optimal working hours
        if 'Hours_Per_Week' in self.processed_df.columns:
            optimal_hours_df = self.processed_df[(self.processed_df['Hours_Per_Week'] >= 35) &
                                               (self.processed_df['Hours_Per_Week'] <= 45)]
            if len(optimal_hours_df) > 0:
                healthy_percentage = (optimal_hours_df['Mental_Health_Status'] == 'Healthy').mean() * 100
                protective_factors.append({
                    'factor': 'Optimal Working Hours',
                    'threshold': '35-45 hours/week',
                    'benefited_population': f'{len(optimal_hours_df)} employees ({len(optimal_hours_df)/len(self.processed_df)*100:.1f}%)',
                    'protection_level': float(healthy_percentage),
                    'description': f'{healthy_percentage:.1f}% of employees with optimal hours maintain good mental health'
                })

        return protective_factors

    def _analyze_population_segments(self):
        """Analyze different population segments"""
        segments = []

        # By work arrangement
        if 'Work_Arrangement' in self.processed_df.columns:
            for arrangement in self.processed_df['Work_Arrangement'].unique():
                arr_data = self.processed_df[self.processed_df['Work_Arrangement'] == arrangement]
                healthy_pct = (arr_data['Mental_Health_Status'] == 'Healthy').mean() * 100
                avg_wlb = arr_data['Work_Life_Balance_Score'].mean() if 'Work_Life_Balance_Score' in arr_data.columns else 0

                segments.append({
                    'segment': f'{arrangement} Workers',
                    'size': len(arr_data),
                    'healthy_percentage': float(healthy_pct),
                    'avg_wlb_score': float(avg_wlb),
                    'key_characteristic': f'{healthy_pct:.1f}% maintain healthy mental status'
                })

        # By age groups
        if 'Age' in self.processed_df.columns:
            age_groups = pd.cut(self.processed_df['Age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-39', '40-49', '50+'])
            for age_group in age_groups.unique():
                if pd.notna(age_group):
                    group_data = self.processed_df[age_groups == age_group]
                    healthy_pct = (group_data['Mental_Health_Status'] == 'Healthy').mean() * 100

                    segments.append({
                        'segment': f'Age {age_group}',
                        'size': len(group_data),
                        'healthy_percentage': float(healthy_pct),
                        'avg_wlb_score': float(group_data['Work_Life_Balance_Score'].mean() if 'Work_Life_Balance_Score' in group_data.columns else 0),
                        'key_characteristic': f'{healthy_pct:.1f}% healthy rate'
                    })

        return segments

    def _extract_key_findings(self, feature_importance):
        """Extract key actionable findings"""
        findings = []

        # Top predictor analysis
        top_predictor = max(feature_importance.items(), key=lambda x: x[1])
        findings.append({
            'type': 'Primary Driver',
            'finding': f'{top_predictor[0]} is the strongest predictor of mental health outcomes',
            'impact': float(top_predictor[1] * 100),
            'action': 'Prioritize interventions targeting this factor for maximum impact'
        })

        # Population at risk
        total_at_risk = 0
        if 'Mental_Health_Status' in self.processed_df.columns:
            at_risk_conditions = ['Stress Disorder', 'Depression', 'Anxiety']
            total_at_risk = self.processed_df['Mental_Health_Status'].isin(at_risk_conditions).sum()

        findings.append({
            'type': 'Population Risk',
            'finding': f'{total_at_risk} employees ({total_at_risk/len(self.processed_df)*100:.1f}%) show concerning mental health symptoms',
            'impact': float(total_at_risk/len(self.processed_df)*100),
            'action': 'Implement immediate support programs for high-risk employees'
        })

        # Work-life balance insight
        if 'Work_Life_Balance_Score' in self.processed_df.columns:
            avg_wlb = self.processed_df['Work_Life_Balance_Score'].mean()
            findings.append({
                'type': 'Organizational Health',
                'finding': f'Average work-life balance score is {avg_wlb:.1f}/10 - {"Good" if avg_wlb >= 7 else "Needs Improvement" if avg_wlb >= 5 else "Critical"}',
                'impact': float(avg_wlb * 10),
                'action': 'Focus on flexible work policies and time management training' if avg_wlb < 7 else 'Maintain current positive work-life balance initiatives'
            })

        return findings

    def clustering_analysis(self, n_clusters=None, selected_features=None):
        """Enhanced clustering analysis"""
        if self.processed_df is None:
            self.preprocess_data()

        # Default features for clustering
        default_features = ['Age', 'Hours_Per_Week', 'Work_Life_Balance_Score', 'Social_Isolation_Score']
        cluster_features = selected_features or default_features
        cluster_features = [col for col in cluster_features if col in self.processed_df.columns]

        if len(cluster_features) < 2:
            return None

        X_cluster = self.processed_df[cluster_features].fillna(0)
        X_cluster_scaled = self.scaler.fit_transform(X_cluster)

        # Determine optimal number of clusters if not provided
        if n_clusters is None:
            inertias = []
            silhouette_scores = []
            K_range = range(2, min(8, len(X_cluster) // 10))

            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_cluster_scaled)
                inertias.append(float(kmeans.inertia_))

                from sklearn.metrics import silhouette_score
                score = silhouette_score(X_cluster_scaled, kmeans.labels_)
                silhouette_scores.append(float(score))

            optimal_k = list(K_range)[np.argmax(silhouette_scores)] if silhouette_scores else 3
        else:
            optimal_k = n_clusters
            inertias = []
            silhouette_scores = []
            K_range = range(2, min(8, len(X_cluster) // 10))

        # Final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_cluster_scaled)

        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_cluster_scaled)

        # Add clusters to dataframe
        cluster_df = self.processed_df.copy()
        cluster_df['Cluster'] = clusters
        cluster_df['PCA_1'] = X_pca[:, 0]
        cluster_df['PCA_2'] = X_pca[:, 1]

        # Analyze clusters
        cluster_summary = {}
        for i in range(optimal_k):
            cluster_data = cluster_df[cluster_df['Cluster'] == i]
            cluster_summary[f'Cluster_{i}'] = {
                'size': int(len(cluster_data)),
                'avg_age': float(cluster_data['Age'].mean()) if 'Age' in cluster_data.columns else 0.0,
                'avg_hours': float(
                    cluster_data['Hours_Per_Week'].mean()) if 'Hours_Per_Week' in cluster_data.columns else 0.0,
                'avg_work_life_balance': float(cluster_data[
                                                   'Work_Life_Balance_Score'].mean()) if 'Work_Life_Balance_Score' in cluster_data.columns else 0.0,
                'avg_isolation': float(cluster_data[
                                           'Social_Isolation_Score'].mean()) if 'Social_Isolation_Score' in cluster_data.columns else 0.0,
                'mental_health_dist': {str(k): int(v) for k, v in cluster_data[
                    'Mental_Health_Status'].value_counts().to_dict().items()} if 'Mental_Health_Status' in cluster_data.columns else {}
            }

        return {
            'optimal_k': int(optimal_k),
            'clusters': [int(c) for c in clusters],
            'cluster_summary': cluster_summary,
            'silhouette_scores': [(int(k), float(score)) for k, score in zip(K_range, silhouette_scores)],
            'pca_data': {
                'x': X_pca[:, 0].tolist(),
                'y': X_pca[:, 1].tolist(),
                'clusters': [int(c) for c in clusters],
                'explained_variance': [float(pca.explained_variance_ratio_[0]), float(pca.explained_variance_ratio_[1])]
            },
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'feature_names': cluster_features
        }


def load_data():
    """Load and validate data with fingerprinting"""
    global df, data_fingerprint
    try:
        # Kiểm tra nhiều vị trí có thể
        possible_paths = [
            'data/data.csv',  # Trong folder data
            'data.csv',  # Trong root
            './data/data.csv',  # Path tương đối
        ]

        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break

        if not data_path:
            return False, "Data file not found. Please ensure data.csv is in the data directory or root directory"

        df = pd.read_csv(data_path)
        generate_data_fingerprint()

        print(f"Data loaded successfully from {data_path}: {len(df)} rows, {len(df.columns)} columns")
        print(f"Data fingerprint: {data_fingerprint}")

        if df.empty:
            return False, "CSV file is empty"
        if len(df.columns) < 5:
            return False, "CSV file doesn't have enough columns"

        return True, f"Successfully loaded {len(df)} records with {len(df.columns)} columns from {data_path}"

    except FileNotFoundError:
        return False, "data.csv file not found"
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False, f"Error reading file: {str(e)}"


def create_all_visualizations(filters=None):
    """Create all visualizations with optional filtering"""
    if df is None:
        return {}

    # Apply filters if provided
    filtered_df = df.copy()
    if filters:
        for key, value in filters.items():
            if key in filtered_df.columns and value and value != 'all':
                filtered_df = filtered_df[filtered_df[key] == value]

    plots = {}

    try:
        # 1. Dashboard Overview
        fig_overview = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Mental Health Distribution', 'Burnout by Industry',
                            'Work-Life Balance', 'Working Hours'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "box"}, {"type": "histogram"}]]
        )

        if 'Mental_Health_Status' in filtered_df.columns:
            mental_health_counts = filtered_df['Mental_Health_Status'].value_counts()
            fig_overview.add_trace(
                go.Pie(labels=mental_health_counts.index, values=mental_health_counts.values, name="Mental Health"),
                row=1, col=1
            )

        if 'Industry' in filtered_df.columns and 'Burnout_Level' in filtered_df.columns:
            burnout_counts = filtered_df.groupby(['Industry', 'Burnout_Level']).size().reset_index(name='count')
            for burnout in filtered_df['Burnout_Level'].unique():
                data = burnout_counts[burnout_counts['Burnout_Level'] == burnout]
                fig_overview.add_trace(
                    go.Bar(x=data['Industry'], y=data['count'], name=f'Burnout: {burnout}'),
                    row=1, col=2
                )

        if 'Work_Life_Balance_Score' in filtered_df.columns and 'Mental_Health_Status' in filtered_df.columns:
            for status in filtered_df['Mental_Health_Status'].unique():
                data = filtered_df[filtered_df['Mental_Health_Status'] == status]['Work_Life_Balance_Score']
                fig_overview.add_trace(
                    go.Box(y=data, name=status),
                    row=2, col=1
                )

        if 'Hours_Per_Week' in filtered_df.columns:
            fig_overview.add_trace(
                go.Histogram(x=filtered_df['Hours_Per_Week'], name="Hours/Week", nbinsx=20),
                row=2, col=2
            )

        fig_overview.update_layout(height=800, showlegend=True, title_text="Dashboard Overview")
        plots['dashboard_overview'] = json.dumps(fig_overview, cls=plotly.utils.PlotlyJSONEncoder)

        # 2. 3D Scatter Plot
        if all(col in filtered_df.columns for col in
               ['Age', 'Hours_Per_Week', 'Work_Life_Balance_Score', 'Mental_Health_Status']):
            fig_3d = px.scatter_3d(filtered_df,
                                   x='Age',
                                   y='Hours_Per_Week',
                                   z='Work_Life_Balance_Score',
                                   color='Mental_Health_Status',
                                   title='3D Analysis: Age - Working Hours - Work-Life Balance',
                                   labels={'Age': 'Age', 'Hours_Per_Week': 'Hours/week',
                                           'Work_Life_Balance_Score': 'WLB Score'})
            plots['scatter_3d'] = json.dumps(fig_3d, cls=plotly.utils.PlotlyJSONEncoder)

        # 3. Correlation Heatmap
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 3:
            correlation_matrix = filtered_df[numeric_cols].corr()
            fig_heatmap = px.imshow(correlation_matrix,
                                    labels=dict(color="Correlation"),
                                    title="Detailed Correlation Matrix",
                                    color_continuous_scale='RdBu_r')
            fig_heatmap.update_layout(width=800, height=600)
            plots['correlation_heatmap'] = json.dumps(fig_heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        # 4. Sunburst Chart
        if all(col in filtered_df.columns for col in ['Region', 'Industry', 'Mental_Health_Status']):
            fig_sunburst = px.sunburst(filtered_df,
                                       path=['Region', 'Industry', 'Mental_Health_Status'],
                                       title='Multi-level Distribution: Region - Industry - Mental Health')
            plots['sunburst'] = json.dumps(fig_sunburst, cls=plotly.utils.PlotlyJSONEncoder)

        # 5. Violin plots
        if all(col in filtered_df.columns for col in ['Work_Arrangement', 'Social_Isolation_Score']):
            fig_violin = px.violin(filtered_df,
                                   x='Work_Arrangement',
                                   y='Social_Isolation_Score',
                                   color='Work_Arrangement',
                                   title='Isolation Level by Work Arrangement')
            plots['violin_arrangement'] = json.dumps(fig_violin, cls=plotly.utils.PlotlyJSONEncoder)

    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")

    return plots


def perform_statistical_tests(var1=None, var2=None):
    """Perform statistical tests with detailed results"""
    if df is None:
        return {}

    results = {}

    try:
        # Chi-square test
        if 'Gender' in df.columns and 'Mental_Health_Status' in df.columns:
            contingency_table = pd.crosstab(df['Gender'], df['Mental_Health_Status'])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            results['chi_square_gender_mental'] = {
                'test_name': 'Chi-square: Gender vs Mental Health',
                'hypothesis_h0': 'No association between gender and mental health status',
                'hypothesis_h1': 'There is an association between gender and mental health status',
                'statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'alpha': 0.05,
                'interpretation': 'Significant association found' if p_value < 0.05 else 'No significant association',
                'effect_size': float(
                    np.sqrt(chi2 / (contingency_table.sum().sum() * (min(contingency_table.shape) - 1)))),
                'contingency_table': contingency_table.to_dict()
            }

        # T-test
        if 'Hours_Per_Week' in df.columns and 'Mental_Health_Status' in df.columns:
            mental_categories = df['Mental_Health_Status'].unique()
            if len(mental_categories) >= 2:
                group1 = df[df['Mental_Health_Status'] == mental_categories[0]]['Hours_Per_Week'].dropna()
                group2 = df[df['Mental_Health_Status'] == mental_categories[1]]['Hours_Per_Week'].dropna()

                t_stat, p_value = ttest_ind(group1, group2)
                cohens_d = (group1.mean() - group2.mean()) / np.sqrt(
                    ((len(group1) - 1) * group1.var() + (len(group2) - 1) * group2.var()) / (
                                len(group1) + len(group2) - 2))

                results['t_test_hours_mental'] = {
                    'test_name': f'T-test: Working hours between {mental_categories[0]} and {mental_categories[1]}',
                    'hypothesis_h0': 'No difference in mean working hours between groups',
                    'hypothesis_h1': 'There is a difference in mean working hours between groups',
                    'statistic': float(t_stat),
                    'p_value': float(p_value),
                    'alpha': 0.05,
                    'interpretation': 'Significant difference found' if p_value < 0.05 else 'No significant difference',
                    'effect_size_cohens_d': float(cohens_d),
                    'group1_mean': float(group1.mean()),
                    'group2_mean': float(group2.mean()),
                    'group1_size': int(len(group1)),
                    'group2_size': int(len(group2))
                }

        # ANOVA
        if 'Work_Life_Balance_Score' in df.columns and 'Mental_Health_Status' in df.columns:
            groups = [group['Work_Life_Balance_Score'].dropna() for name, group in df.groupby('Mental_Health_Status')]
            if len(groups) >= 2:
                f_stat, p_value = f_oneway(*groups)

                # Calculate eta-squared (effect size for ANOVA)
                total_mean = df['Work_Life_Balance_Score'].mean()
                ss_total = ((df['Work_Life_Balance_Score'] - total_mean) ** 2).sum()
                ss_between = sum(len(group) * (group.mean() - total_mean) ** 2 for group in groups)
                eta_squared = ss_between / ss_total if ss_total != 0 else 0

                results['anova_wlb_mental'] = {
                    'test_name': 'ANOVA: Work-Life Balance across mental health groups',
                    'hypothesis_h0': 'No difference in mean Work-Life Balance across groups',
                    'hypothesis_h1': 'At least one group has different mean Work-Life Balance',
                    'statistic': float(f_stat),
                    'p_value': float(p_value),
                    'alpha': 0.05,
                    'interpretation': 'Significant difference between groups' if p_value < 0.05 else 'No significant difference',
                    'effect_size_eta_squared': float(eta_squared),
                    'groups_count': len(groups),
                    'group_means': {f'Group_{i}': float(group.mean()) for i, group in enumerate(groups)}
                }

    except Exception as e:
        results['error'] = f'Error performing tests: {str(e)}'

    return results


def generate_recommendations(analysis_result):
    """Generate actionable recommendations"""
    recommendations = []

    if 'feature_importance' in analysis_result:
        best_model = analysis_result['best_model']
        feature_importance = analysis_result['feature_importance'][best_model]

        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        # Generate specific recommendations based on top features
        top_features = [item[0] for item in sorted_features[:3]]

        for i, (feature, importance) in enumerate(sorted_features[:5]):
            priority = 'High' if i < 2 else 'Medium'

            if 'Work_Life_Balance' in feature:
                recommendations.append({
                    'priority': priority,
                    'area': 'Work-Life Balance',
                    'recommendation': 'Implement flexible working hours and encourage employees to use vacation time',
                    'expected_impact': '25-30% improvement in mental health status',
                    'action_items': [
                        'Review current work hour policies',
                        'Implement flexible start/end times',
                        'Monitor overtime patterns',
                        'Create vacation usage incentives'
                    ],
                    'kpis': ['Work-Life Balance Score', 'Overtime Hours', 'Vacation Days Used'],
                    'timeline': '3-6 months'
                })

            elif 'Social_Isolation' in feature:
                recommendations.append({
                    'priority': priority,
                    'area': 'Social Interaction',
                    'recommendation': 'Organize regular team building activities and create collaborative workspaces',
                    'expected_impact': '20-25% reduction in social isolation',
                    'action_items': [
                        'Schedule monthly team events',
                        'Create shared collaboration spaces',
                        'Implement buddy system for new employees',
                        'Establish cross-functional project teams'
                    ],
                    'kpis': ['Social Isolation Score', 'Team Collaboration Rating', 'Employee Engagement'],
                    'timeline': '2-4 months'
                })

            elif 'Hours_Per_Week' in feature:
                recommendations.append({
                    'priority': priority,
                    'area': 'Workload Management',
                    'recommendation': 'Monitor and control working hours to prevent chronic overwork',
                    'expected_impact': '15-20% improvement in burnout levels',
                    'action_items': [
                        'Set maximum weekly hour limits',
                        'Implement workload distribution tools',
                        'Regular manager check-ins on workload',
                        'Hire additional staff if needed'
                    ],
                    'kpis': ['Average Weekly Hours', 'Burnout Level', 'Productivity Metrics'],
                    'timeline': '1-3 months'
                })

    return recommendations


# API Routes

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/load_data')
def api_load_data():
    try:
        success, message = load_data()
        if success:
            basic_stats = {
                'total_records': int(len(df)),
                'total_columns': int(len(df.columns)),
                'missing_values': int(df.isnull().sum().sum()),
                'data_types': {str(k): int(v) for k, v in df.dtypes.value_counts().to_dict().items()},
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB"
            }

            # Schema information for UI
            schema_info = {
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'column_info': {col: {
                    'dtype': str(df[col].dtype),
                    'unique_count': int(df[col].nunique()),
                    'null_count': int(df[col].isnull().sum()),
                    'sample_values': df[col].dropna().unique()[:10].tolist() if df[col].dtype == 'object' else None
                } for col in df.columns}
            }

            sample_data = df.head().copy()
            for col in sample_data.columns:
                sample_data[col] = sample_data[col].astype(str)
            sample_data = sample_data.fillna('N/A')

            response_data = {
                'status': 'success',
                'message': message,
                'basic_stats': basic_stats,
                'schema_info': schema_info,
                'columns': list(df.columns),
                'sample_data': sample_data.to_dict('records'),
                'data_fingerprint': data_fingerprint
            }

            return jsonify(response_data)
        else:
            return jsonify({'status': 'error', 'message': message})

    except Exception as e:
        print(f"Error in api_load_data: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Server error: {str(e)}'})


@app.route('/api/visualizations')
def api_visualizations():
    if df is None:
        return jsonify({'status': 'error', 'message': 'No data loaded'})

    try:
        # Get filter parameters
        filters = {}
        for param in ['Region', 'Industry', 'Mental_Health_Status']:
            value = request.args.get(param.lower())
            if value and value != 'all':
                filters[param] = value

        cache_key = f"viz_{data_fingerprint}_{hash(str(sorted(filters.items())))}"

        if cache_key not in analysis_cache:
            analysis_cache[cache_key] = create_all_visualizations(filters)

        plots = analysis_cache[cache_key]
        return jsonify({'status': 'success', 'plots': plots, 'filters_applied': filters})

    except Exception as e:
        print(f"Error in api_visualizations: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Visualization error: {str(e)}'})


@app.route('/api/ml_analysis')
def api_ml_analysis():
    if df is None:
        return jsonify({'status': 'error', 'message': 'No data loaded'})

    try:
        cache_key = f"ml_{data_fingerprint}"

        if cache_key not in analysis_cache:
            analyzer = DataAnalyzer(df)
            analysis_cache[cache_key] = analyzer.advanced_feature_analysis()

        result = analysis_cache[cache_key]

        if result is None:
            return jsonify({'status': 'error', 'message': 'Cannot perform ML analysis'})

        # Return improved ML analysis with insights
        formatted_result = {
            'status': 'success',
            'feature_importance': result['feature_importance'],
            'data_insights': result['data_insights'],
            'risk_factors': result['risk_factors'],
            'protective_factors': result['protective_factors'],
            'population_segments': result['population_segments'],
            'key_findings': result['key_findings'],
            'target_classes': result['target_classes'],
            'feature_names': result['feature_names'],
            'sample_size': result['sample_size'],
            'analysis_date': result['analysis_date']
        }

        return jsonify(formatted_result)

    except Exception as e:
        print(f"Error in api_ml_analysis: {str(e)}")
        return jsonify({'status': 'error', 'message': f'ML analysis error: {str(e)}'})


@app.route('/api/clustering')
def api_clustering():
    if df is None:
        return jsonify({'status': 'error', 'message': 'No data loaded'})

    try:
        # Get parameters
        n_clusters = request.args.get('k', type=int)
        selected_features = request.args.getlist('features')

        cache_key = f"cluster_{data_fingerprint}_{n_clusters}_{hash(str(sorted(selected_features)))}"

        if cache_key not in analysis_cache:
            analyzer = DataAnalyzer(df)
            analysis_cache[cache_key] = analyzer.clustering_analysis(n_clusters, selected_features)

        result = analysis_cache[cache_key]

        if result is None:
            return jsonify({'status': 'error', 'message': 'Cannot perform clustering analysis'})

        return jsonify({
            'status': 'success',
            'clustering_results': result
        })

    except Exception as e:
        print(f"Error in api_clustering: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Clustering error: {str(e)}'})


@app.route('/api/statistical_tests')
def api_statistical_tests():
    if df is None:
        return jsonify({'status': 'error', 'message': 'No data loaded'})

    try:
        var1 = request.args.get('var1')
        var2 = request.args.get('var2')

        cache_key = f"stats_{data_fingerprint}_{var1}_{var2}"

        if cache_key not in analysis_cache:
            analysis_cache[cache_key] = perform_statistical_tests(var1, var2)

        tests_results = analysis_cache[cache_key]
        return jsonify({'status': 'success', 'tests': tests_results})

    except Exception as e:
        print(f"Error in api_statistical_tests: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Statistical test error: {str(e)}'})


@app.route('/api/initial_load')
def api_initial_load():
    """API endpoint for automatic initial data loading"""
    try:
        success, message = load_data()
        if success:
            basic_stats = {
                'total_records': int(len(df)),
                'total_columns': int(len(df.columns)),
                'missing_values': int(df.isnull().sum().sum()),
                'data_types': {str(k): int(v) for k, v in df.dtypes.value_counts().to_dict().items()},
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB"
            }

            # Schema information
            schema_info = {
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
                'column_info': {col: {
                    'dtype': str(df[col].dtype),
                    'unique_count': int(df[col].nunique()),
                    'null_count': int(df[col].isnull().sum()),
                    'sample_values': df[col].dropna().unique()[:10].tolist() if df[col].dtype == 'object' else None
                } for col in df.columns}
            }

            sample_data = df.head().copy()
            for col in sample_data.columns:
                sample_data[col] = sample_data[col].astype(str)
            sample_data = sample_data.fillna('N/A')

            # Create initial visualizations
            initial_plots = create_all_visualizations()

            response_data = {
                'status': 'success',
                'message': message,
                'basic_stats': basic_stats,
                'schema_info': schema_info,
                'columns': list(df.columns),
                'sample_data': sample_data.to_dict('records'),
                'initial_plots': initial_plots,
                'data_fingerprint': data_fingerprint
            }

            return jsonify(response_data)
        else:
            return jsonify({'status': 'error', 'message': message})

    except Exception as e:
        print(f"Error in api_initial_load: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Server error: {str(e)}'})


def initialize_data():
    """Automatically load data when server starts"""
    success, message = load_data()
    if success:
        print(f"✅ Auto-loaded data on server start: {message}")
    else:
        print(f"⚠️ Could not auto-load data: {message}")


@app.route('/api/recommendations')
def api_recommendations():
    if df is None:
        return jsonify({'status': 'error', 'message': 'No data loaded'})

    try:
        # Tận dụng cache ML đã tính
        cache_key_ml = f"ml_{data_fingerprint}"
        if cache_key_ml not in analysis_cache:
            analyzer = DataAnalyzer(df)
            analysis_cache[cache_key_ml] = analyzer.advanced_feature_analysis()
        ml_result = analysis_cache[cache_key_ml]
        if ml_result is None:
            return jsonify({'status': 'error', 'message': 'Cannot perform ML analysis'})

        # 1) Gợi ý “ngắn” từ ML backend sẵn có
        short_recs = generate_recommendations(ml_result)  # đã có sẵn trong code backend

        # 2) Gợi ý “đầy đủ” theo cấu trúc frontend đang render
        #    Ta sử dụng feature importance & data thực từ backend để tạo các khối high/medium/strategic...
        #    (Sao chép logic từ Recommendations.generateComprehensiveRecommendations phía FE)
        # Chuẩn bị đầu vào
        best_model = ml_result['best_model']
        feat_imp = ml_result['feature_importance'][best_model] if best_model in ml_result['feature_importance'] else {}
        top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:5]

        # Phân tích pattern nhẹ từ df (giống analyzeDataPatterns ở FE)
        patterns = {
            'mentalHealthDistribution': df['Mental_Health_Status'].value_counts().to_dict()
                if 'Mental_Health_Status' in df.columns else {},
            'industryTrends': df['Industry'].value_counts().to_dict()
                if 'Industry' in df.columns else {},
            'workArrangementImpact': {},
            'burnoutPrevalence': df['Burnout_Level'].value_counts().to_dict()
                if 'Burnout_Level' in df.columns else {}
        }
        if {'Work_Arrangement','Work_Life_Balance_Score'}.issubset(df.columns):
            wa = (df.groupby('Work_Arrangement')['Work_Life_Balance_Score']
                    .agg(['count','mean']).reset_index())
            for _, r in wa.iterrows():
                patterns['workArrangementImpact'][str(r['Work_Arrangement'])] = {
                    'count': int(r['count']),
                    'avgWLB': float(r['mean'])
                }

        # Các factory hàm sinh recs dài hạn (copy tinh gọn từ FE)
        def high_priority(top_features):
            out = []
            for idx, (f, imp) in enumerate(top_features):
                if imp > 0.15 and 'Work_Life_Balance' in f:
                    out.append({
                        'id': f'high_wlb_{idx}',
                        'title': 'Implement Flexible Work-Life Balance Program',
                        'description': 'Work-life balance is a critical predictor of mental health outcomes.',
                        'impact': 'High','effort':'Medium','timeline':'1-3 months',
                        'expectedImprovement':'25-35%','kpis':['Work-Life Balance Score','Employee Satisfaction','Turnover Rate'],
                        'cost':'Low-Medium','department':['HR','Operations'],
                        'evidence': f'Feature importance: {imp*100:.1f}%'
                    })
                if imp > 0.15 and 'Social_Isolation' in f:
                    out.append({
                        'id': f'high_social_{idx}',
                        'title': 'Combat Social Isolation with Connection Programs',
                        'description': 'Social isolation significantly impacts mental health.',
                        'impact': 'High','effort':'Medium','timeline':'2-4 months',
                        'expectedImprovement':'30-40%','kpis':['Social Isolation Score','Team Collaboration Rating','Employee Engagement'],
                        'cost':'Low','department':['HR','IT','Facilities'],
                        'evidence': f'Feature importance: {imp*100:.1f}%'
                    })
                if imp > 0.15 and 'Hours_Per_Week' in f:
                    out.append({
                        'id': f'high_hours_{idx}',
                        'title': 'Workload Management and Hour Monitoring System',
                        'description':'Excessive working hours are strongly linked to mental health issues.',
                        'impact':'High','effort':'Low','timeline':'1-2 months',
                        'expectedImprovement':'20-30%','kpis':['Average Weekly Hours','Overtime Frequency','Productivity per Hour'],
                        'cost':'Low','department':['HR','Management'],
                        'evidence': f'Feature importance: {imp*100:.1f}%'
                    })
            return out

        def medium_priority(patterns):
            out = []
            # Industry programs (top-3 ngành)
            ind = sorted(patterns.get('industryTrends',{}).items(), key=lambda x:x[1], reverse=True)[:3]
            for industry, cnt in ind:
                out.append({
                    'id': f'med_industry_{industry}',
                    'title': f'Industry-Specific Mental Health Program for {industry}',
                    'description': f'Tailored interventions for {industry}.',
                    'impact':'Medium','effort':'Medium','timeline':'3-6 months',
                    'expectedImprovement':'15-25%','kpis':['Industry Satisfaction Score','Sector-specific Burnout Rates','Program Engagement'],
                    'cost':'Medium','department':['HR','External Partners'],
                    'evidence': f'{cnt} employees in {industry} sector'
                })
            # Work arrangement tối ưu
            wai = patterns.get('workArrangementImpact',{})
            if wai:
                best = max(wai.items(), key=lambda x: x[1].get('avgWLB',0.0))
                out.append({
                    'id':'med_work_arrangement',
                    'title':'Optimize Work Arrangement Policies',
                    'description': f'{best[0]} shows best work-life balance results. Consider expanding this model.',
                    'impact':'Medium','effort':'High','timeline':'4-8 months',
                    'expectedImprovement':'20-30%','kpis':['Work Arrangement Satisfaction','Productivity Metrics','Employee Retention'],
                    'cost':'Medium-High','department':['HR','IT','Facilities'],
                    'evidence': f'{best[0]} highest WLB ({best[1].get("avgWLB",0):.1f})'
                })
            return out

        def strategic():
            return [
                {'id':'strategic_culture','title':'Comprehensive Mental Health Culture Transformation',
                 'description':'Long-term culture change to prioritize mental health.',
                 'impact':'Very High','effort':'Very High','timeline':'12-24 months',
                 'expectedImprovement':'40-60%','kpis':['Overall Employee Wellbeing Index','Culture Survey Scores','Leadership Engagement'],
                 'cost':'High','department':['Executive','HR','All Departments']},
                {'id':'strategic_data','title':'Advanced Mental Health Analytics Platform',
                 'description':'Predictive analytics and AI-driven monitoring.',
                 'impact':'High','effort':'High','timeline':'8-12 months',
                 'expectedImprovement':'25-40%','kpis':['Prediction Accuracy','Early Intervention Success','Data Usage Metrics'],
                 'cost':'High','department':['IT','HR','Data Analytics']}
            ]

        def implementation():
            return {
                'phases': [
                    {'phase':'Phase 1: Foundation (Months 1-3)','description':'Establish basic support',
                     'milestones': ['Needs assessment','EAP launch','Basic WLB policies','Manager training','Baseline metrics'],
                     'resources':'HR + Consultants (~$50k-100k)','risks':'Buy-in / management support'},
                    {'phase':'Phase 2: Development (Months 4-8)','description':'Pilot & develop programs',
                     'milestones': ['Pilot programs','Workload systems','Connection initiatives','Resource library','Advanced analytics start'],
                     'resources':'Cross-functional + IT (~$100k-200k)','risks':'Scalability / integration'},
                    {'phase':'Phase 3: Scale & Optimize (Months 9-12)','description':'Scale org-wide',
                     'milestones': ['Org rollout','Monitoring systems','Continuous improvement','Partnerships','ROI reporting'],
                     'resources':'Org-wide (~$150k-300k)','risks':'Change resistance / sustainability'}
                ],
                'criticalSuccessFactors': [
                    'Executive sponsorship','Employee co-design','Process integration','Continuous measurement','Adequate funding'
                ],
                'riskMitigation': [
                    'Transparent comms','Phased rollout + feedback','Training & change mgmt','Governance & accountability','Budget contingency'
                ]
            }

        def cost_benefit_and_roi():
            costs = {
                'year1': {'Total': 305000},
                'year2': {'Total': 290000},
                'year3': {'Total': 325000},
            }
            bens  = {
                'year1': {'Total': 415000},
                'year2': {'Total': 590000},
                'year3': {'Total': 795000},
            }
            yearly = []
            cum_inv = cum_ben = 0
            for y in (1,2,3):
                inv = costs[f'year{y}']['Total']; ben = bens[f'year{y}']['Total']
                yearly.append({
                    'year': y, 'investment': inv, 'benefits': ben,
                    'netReturn': ben - inv, 'roiPercentage': f"{(ben-inv)/inv*100:.1f}"
                })
                cum_inv += inv; cum_ben += ben
            cumulative = {
                'totalInvestment': cum_inv, 'totalBenefits': cum_ben,
                'netReturn': cum_ben - cum_inv,
                'roiPercentage': f"{(cum_ben-cum_inv)/cum_inv*100:.1f}"
            }
            return {
                'costBenefit': {'costs': costs, 'benefits': bens},
                'roi': {'yearlyROI': yearly, 'cumulativeROI': cumulative, 'paybackPeriod': '14 months',
                        'assumptions': ['Turnover cost ~ $50k/employee','Productivity +15–25%','Healthcare cost ↓','Satisfaction ↑ → recruitment cost ↓']}
            }

        # Gộp kết quả
        cbroi = cost_benefit_and_roi()
        full = {
            'high': high_priority(top_features),
            'medium': medium_priority(patterns),
            'strategic': strategic(),
            'implementation': implementation(),
            'costBenefit': cbroi['costBenefit'],
            'roi': cbroi['roi'],
            'kpis': {
                'primary': [
                    {'name':'Employee Mental Health Index','target':'85% positive','frequency':'Monthly','owner':'HR'},
                    {'name':'Work-Life Balance Score','target':'≥7.5/10','frequency':'Quarterly','owner':'Ops'}
                ],
                'secondary': [
                    {'name':'Employee Turnover Rate','target':'<5%/year','frequency':'Monthly','owner':'HR Analytics'}
                ],
                'operational': [
                    {'name':'Average Weekly Working Hours','target':'<45h','frequency':'Weekly','owner':'Managers'}
                ]
            }
        }

        # Priority matrix
        def to_matrix(recs):
            impact = {'Low':1,'Medium':2,'High':3,'Very High':4}
            effort = {'Low':1,'Medium':2,'High':3,'Very High':4}
            matrix = []
            for cat in ['high','medium','strategic']:
                for r in recs.get(cat, []):
                    matrix.append({
                        'id': r['id'], 'title': r['title'],
                        'impact': impact.get(r.get('impact','Medium'),2),
                        'effort': effort.get(r.get('effort','Medium'),2),
                        'category': cat, 'expectedImprovement': r.get('expectedImprovement'), 'timeline': r.get('timeline')
                    })
            return matrix
        full['priorityMatrix'] = to_matrix(full)

        return jsonify({'status':'success', 'short_recommendations': short_recs, 'recommendations': full})

    except Exception as e:
        print(f"Error in api_recommendations: {str(e)}")
        return jsonify({'status':'error','message':f'Recommendations error: {str(e)}'})


@app.route('/api/work_arrangement_analysis')
def api_work_arrangement_analysis():
    """Work arrangement analysis endpoint"""
    if df is None:
        return jsonify({'status': 'error', 'message': 'No data loaded'})

    try:
        cache_key = f"work_arrangement_{data_fingerprint}"

        if cache_key not in analysis_cache:
            # Analyze work arrangement patterns
            analysis_result = {}

            if 'Work_Arrangement' in df.columns:
                # Mental health distribution by work arrangement
                if 'Mental_Health_Status' in df.columns:
                    mental_health_crosstab = pd.crosstab(df['Work_Arrangement'], df['Mental_Health_Status'], normalize='index') * 100
                    mental_health_distribution = {}
                    for arrangement in mental_health_crosstab.index:
                        mental_health_distribution[arrangement] = mental_health_crosstab.loc[arrangement].to_dict()
                    analysis_result['mental_health_distribution'] = mental_health_distribution

                # Stress and burnout analysis
                stress_burnout_analysis = {}
                for arrangement in df['Work_Arrangement'].unique():
                    arr_data = df[df['Work_Arrangement'] == arrangement]
                    stress_burnout_analysis[arrangement] = {
                        'avg_work_life_balance_score': float(arr_data['Work_Life_Balance_Score'].mean()) if 'Work_Life_Balance_Score' in df.columns else 0,
                        'avg_social_isolation_score': float(arr_data['Social_Isolation_Score'].mean()) if 'Social_Isolation_Score' in df.columns else 0,
                        'avg_hours_per_week': float(arr_data['Hours_Per_Week'].mean()) if 'Hours_Per_Week' in df.columns else 0,
                        'burnout_high_percentage': float((arr_data['Burnout_Level'] == 'High').mean() * 100) if 'Burnout_Level' in df.columns else 0
                    }
                analysis_result['stress_burnout_analysis'] = stress_burnout_analysis

                # Demographic analysis
                demographic_analysis = {}
                for arrangement in df['Work_Arrangement'].unique():
                    arr_data = df[df['Work_Arrangement'] == arrangement]

                    # Gender distribution
                    gender_dist = {}
                    if 'Gender' in df.columns:
                        gender_counts = arr_data['Gender'].value_counts(normalize=True) * 100
                        gender_dist = gender_counts.to_dict()

                    # Age groups
                    age_groups = {}
                    if 'Age' in df.columns:
                        arr_data_copy = arr_data.copy()
                        arr_data_copy['age_group'] = pd.cut(arr_data_copy['Age'],
                                                          bins=[0, 25, 35, 45, 55, 100],
                                                          labels=['18-25', '26-35', '36-45', '46-55', '55+'])
                        age_groups = arr_data_copy['age_group'].value_counts().to_dict()

                    # Top industries
                    industry_top = []
                    if 'Industry' in df.columns:
                        industry_top = arr_data['Industry'].value_counts().head(3).index.tolist()

                    demographic_analysis[arrangement] = {
                        'avg_age': float(arr_data['Age'].mean()) if 'Age' in df.columns else 0,
                        'gender_distribution': gender_dist,
                        'age_groups': {str(k): int(v) for k, v in age_groups.items()},
                        'industry_top': industry_top,
                        'total_employees': int(len(arr_data))
                    }
                analysis_result['demographic_analysis'] = demographic_analysis

                # Productivity metrics (simulated based on available data)
                productivity_metrics = {}
                for arrangement in df['Work_Arrangement'].unique():
                    arr_data = df[df['Work_Arrangement'] == arrangement]

                    # Calculate productivity scores based on available metrics
                    wlb_score = float(arr_data['Work_Life_Balance_Score'].mean()) if 'Work_Life_Balance_Score' in df.columns else 50
                    isolation_score = float(arr_data['Social_Isolation_Score'].mean()) if 'Social_Isolation_Score' in df.columns else 5
                    hours_score = 40 / float(arr_data['Hours_Per_Week'].mean()) * 100 if 'Hours_Per_Week' in df.columns and arr_data['Hours_Per_Week'].mean() > 0 else 50

                    productivity_metrics[arrangement] = {
                        'efficiency': min(100, max(0, hours_score)),
                        'collaboration': min(100, max(0, (10 - isolation_score) * 10)),
                        'innovation': min(100, max(0, wlb_score * 10)),
                        'satisfaction': min(100, max(0, wlb_score * 10)),
                        'retention': min(100, max(0, (10 - isolation_score) * 10))
                    }
                analysis_result['productivity_metrics'] = productivity_metrics

                # Key insights
                insights = []
                if 'Work_Life_Balance_Score' in df.columns:
                    best_wlb_arrangement = df.groupby('Work_Arrangement')['Work_Life_Balance_Score'].mean().idxmax()
                    insights.append(f"{best_wlb_arrangement} work arrangement shows the highest work-life balance scores")

                if 'Social_Isolation_Score' in df.columns:
                    lowest_isolation_arrangement = df.groupby('Work_Arrangement')['Social_Isolation_Score'].mean().idxmin()
                    insights.append(f"{lowest_isolation_arrangement} workers report the lowest social isolation")

                if 'Hours_Per_Week' in df.columns:
                    avg_hours_by_arrangement = df.groupby('Work_Arrangement')['Hours_Per_Week'].mean()
                    lowest_hours = avg_hours_by_arrangement.idxmin()
                    insights.append(f"{lowest_hours} workers have the most reasonable working hours")

                # Recommendations
                recommendations = [
                    "Consider expanding flexible work arrangements that show better work-life balance",
                    "Implement regular check-ins for remote workers to reduce isolation",
                    "Monitor working hours across arrangements to prevent burnout",
                    "Provide arrangement-specific mental health support programs"
                ]

                analysis_result['key_insights'] = insights
                analysis_result['recommendations'] = recommendations

            analysis_cache[cache_key] = analysis_result

        result = analysis_cache[cache_key]
        return jsonify({'status': 'success', 'analysis_results': result})

    except Exception as e:
        print(f"Error in api_work_arrangement_analysis: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Work arrangement analysis error: {str(e)}'})


if __name__ == '__main__':
    initialize_data()
    app.run(debug=True, host='0.0.0.0', port=5000)