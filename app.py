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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables
df = None
models_cache = {}
analysis_cache = {}


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
        """Advanced feature analysis with multiple algorithms"""
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

        # Multiple models comparison
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }

        results = {}
        feature_importance_all = {}

        for name, model in models.items():
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)

            accuracy = accuracy_score(y_test, y_pred)

            results[name] = {
                'accuracy': float(accuracy),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'model': model
            }

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance_all[name] = {col: float(imp) for col, imp in
                                                zip(feature_cols, model.feature_importances_)}
            elif hasattr(model, 'coef_'):
                feature_importance_all[name] = {col: float(imp) for col, imp in
                                                zip(feature_cols, np.abs(model.coef_[0]))}

        return {
            'models_performance': results,
            'feature_importance': feature_importance_all,
            'feature_names': feature_cols,
            'target_encoder': le_target,
            'best_model': max(results.keys(), key=lambda k: results[k]['accuracy'])
        }

    def clustering_analysis(self):
        """Clustering analysis to find employee groups"""
        if self.processed_df is None:
            self.preprocess_data()

        # Select features for clustering
        cluster_features = ['Age', 'Hours_Per_Week', 'Work_Life_Balance_Score', 'Social_Isolation_Score']
        cluster_features = [col for col in cluster_features if col in self.processed_df.columns]

        if len(cluster_features) < 2:
            return None

        X_cluster = self.processed_df[cluster_features].fillna(0)
        X_cluster_scaled = self.scaler.fit_transform(X_cluster)

        # Determine optimal number of clusters
        inertias = []
        silhouette_scores = []
        K_range = range(2, min(8, len(X_cluster) // 10))

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_cluster_scaled)
            inertias.append(float(kmeans.inertia_))

            score = silhouette_score(X_cluster_scaled, kmeans.labels_)
            silhouette_scores.append(float(score))

        # Choose optimal k
        if silhouette_scores:
            optimal_k = list(K_range)[np.argmax(silhouette_scores)]
        else:
            optimal_k = 3

        # Final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_cluster_scaled)

        # Add clusters to dataframe
        cluster_df = self.processed_df.copy()
        cluster_df['Cluster'] = clusters

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
            'cluster_data': cluster_df
        }


def load_data():
    """Load and validate data"""
    global df
    try:
        if not os.path.exists('data.csv'):
            return False, "Không tìm thấy file data.csv trong thư mục hiện tại"

        df = pd.read_csv('data.csv')
        print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")

        if df.empty:
            return False, "File CSV trống"
        if len(df.columns) < 5:
            return False, "File CSV không đủ cột dữ liệu"
        return True, f"Đã tải thành công {len(df)} dòng dữ liệu với {len(df.columns)} cột"
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False, f"Lỗi khi đọc file: {str(e)}"


def create_advanced_visualizations():
    """Create advanced and interactive visualizations"""
    if df is None:
        return {}

    plots = {}

    try:
        # 1. Dashboard Overview
        fig_overview = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Phân bố sức khỏe tinh thần', 'Burnout theo ngành',
                            'Work-Life Balance', 'Số giờ làm việc'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "box"}, {"type": "histogram"}]]
        )

        if 'Mental_Health_Status' in df.columns:
            mental_health_counts = df['Mental_Health_Status'].value_counts()
            fig_overview.add_trace(
                go.Pie(labels=mental_health_counts.index, values=mental_health_counts.values, name="Mental Health"),
                row=1, col=1
            )

        if 'Industry' in df.columns and 'Burnout_Level' in df.columns:
            burnout_counts = df.groupby(['Industry', 'Burnout_Level']).size().reset_index(name='count')
            for burnout in df['Burnout_Level'].unique():
                data = burnout_counts[burnout_counts['Burnout_Level'] == burnout]
                fig_overview.add_trace(
                    go.Bar(x=data['Industry'], y=data['count'], name=f'{burnout}'),
                    row=1, col=2
                )

        if 'Work_Life_Balance_Score' in df.columns and 'Mental_Health_Status' in df.columns:
            for status in df['Mental_Health_Status'].unique():
                data = df[df['Mental_Health_Status'] == status]['Work_Life_Balance_Score']
                fig_overview.add_trace(
                    go.Box(y=data, name=status),
                    row=2, col=1
                )

        if 'Hours_Per_Week' in df.columns:
            fig_overview.add_trace(
                go.Histogram(x=df['Hours_Per_Week'], name="Hours/Week", nbinsx=20),
                row=2, col=2
            )

        fig_overview.update_layout(height=800, showlegend=True, title_text="Dashboard Tổng quan")
        plots['dashboard_overview'] = json.dumps(fig_overview, cls=plotly.utils.PlotlyJSONEncoder)

        # 2. 3D Scatter Plot
        if all(col in df.columns for col in
               ['Age', 'Hours_Per_Week', 'Work_Life_Balance_Score', 'Mental_Health_Status']):
            fig_3d = px.scatter_3d(df,
                                   x='Age',
                                   y='Hours_Per_Week',
                                   z='Work_Life_Balance_Score',
                                   color='Mental_Health_Status',
                                   title='Phân tích 3D: Tuổi - Giờ làm việc - Work-Life Balance',
                                   labels={'Age': 'Tuổi', 'Hours_Per_Week': 'Giờ/tuần',
                                           'Work_Life_Balance_Score': 'Điểm WLB'})
            plots['scatter_3d'] = json.dumps(fig_3d, cls=plotly.utils.PlotlyJSONEncoder)

        # 3. Correlation Heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 3:
            correlation_matrix = df[numeric_cols].corr()
            fig_heatmap = px.imshow(correlation_matrix,
                                    labels=dict(color="Tương quan"),
                                    title="Ma trận tương quan chi tiết",
                                    color_continuous_scale='RdBu_r')
            fig_heatmap.update_layout(width=800, height=600)
            plots['correlation_heatmap'] = json.dumps(fig_heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        # 4. Sunburst Chart
        if all(col in df.columns for col in ['Region', 'Industry', 'Mental_Health_Status']):
            fig_sunburst = px.sunburst(df,
                                       path=['Region', 'Industry', 'Mental_Health_Status'],
                                       title='Phân bố đa cấp: Vùng miền - Ngành nghề - Sức khỏe tinh thần')
            plots['sunburst'] = json.dumps(fig_sunburst, cls=plotly.utils.PlotlyJSONEncoder)

        # 5. Violin plots
        if all(col in df.columns for col in ['Work_Arrangement', 'Social_Isolation_Score']):
            fig_violin = px.violin(df,
                                   x='Work_Arrangement',
                                   y='Social_Isolation_Score',
                                   color='Work_Arrangement',
                                   title='Mức độ cô lập theo hình thức làm việc')
            plots['violin_arrangement'] = json.dumps(fig_violin, cls=plotly.utils.PlotlyJSONEncoder)

    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")

    return plots


def create_clustering_visualization(cluster_result):
    """Create clustering visualization"""
    try:
        cluster_data = cluster_result['cluster_data']

        # 2D projection using PCA
        feature_cols = ['Age', 'Hours_Per_Week', 'Work_Life_Balance_Score', 'Social_Isolation_Score']
        feature_cols = [col for col in feature_cols if col in cluster_data.columns]

        if len(feature_cols) >= 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(cluster_data[feature_cols].fillna(0))

            fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                             color=cluster_data['Cluster'].astype(str),
                             title='Phân nhóm nhân viên (PCA Projection)',
                             labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                                     'y': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'})

            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        print(f"Error creating clustering visualization: {str(e)}")
    return None


def perform_statistical_tests():
    """Perform statistical tests"""
    results = {}

    try:
        # Chi-square test for categorical variables
        if all(col in df.columns for col in ['Gender', 'Mental_Health_Status']):
            contingency_table = pd.crosstab(df['Gender'], df['Mental_Health_Status'])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            results['chi_square_gender_mental'] = {
                'test_name': 'Chi-square: Giới tính vs Sức khỏe tinh thần',
                'statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'interpretation': 'Có mối liên hệ có ý nghĩa' if p_value < 0.05 else 'Không có mối liên hệ có ý nghĩa'
            }

        # T-test for work hours vs mental health
        if all(col in df.columns for col in ['Hours_Per_Week', 'Mental_Health_Status']):
            mental_categories = df['Mental_Health_Status'].unique()
            if len(mental_categories) >= 2:
                group1 = df[df['Mental_Health_Status'] == mental_categories[0]]['Hours_Per_Week'].dropna()
                group2 = df[df['Mental_Health_Status'] == mental_categories[1]]['Hours_Per_Week'].dropna()
                t_stat, p_value = ttest_ind(group1, group2)

                results['t_test_hours_mental'] = {
                    'test_name': f'T-test: Giờ làm việc giữa {mental_categories[0]} và {mental_categories[1]}',
                    'statistic': float(t_stat),
                    'p_value': float(p_value),
                    'interpretation': 'Có sự khác biệt có ý nghĩa' if p_value < 0.05 else 'Không có sự khác biệt có ý nghĩa'
                }

        # ANOVA for work-life balance across mental health categories
        if all(col in df.columns for col in ['Work_Life_Balance_Score', 'Mental_Health_Status']):
            groups = [group['Work_Life_Balance_Score'].dropna() for name, group in df.groupby('Mental_Health_Status')]
            if len(groups) >= 2:
                f_stat, p_value = f_oneway(*groups)

                results['anova_wlb_mental'] = {
                    'test_name': 'ANOVA: Work-Life Balance giữa các nhóm sức khỏe tinh thần',
                    'statistic': float(f_stat),
                    'p_value': float(p_value),
                    'interpretation': 'Có sự khác biệt có ý nghĩa giữa các nhóm' if p_value < 0.05 else 'Không có sự khác biệt có ý nghĩa'
                }

    except Exception as e:
        results['error'] = f'Lỗi thực hiện kiểm định: {str(e)}'

    return results


def generate_recommendations(analysis_result):
    """Generate recommendations based on analysis results"""
    recommendations = []

    if 'feature_importance' in analysis_result:
        best_model = analysis_result['best_model']
        feature_importance = analysis_result['feature_importance'][best_model]

        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        # Generate specific recommendations
        top_feature = sorted_features[0][0] if sorted_features else ''

        if 'Work_Life_Balance' in top_feature:
            recommendations.append({
                'priority': 'Cao',
                'area': 'Cân bằng công việc-cuộc sống',
                'recommendation': 'Thiết lập chính sách linh hoạt về thời gian làm việc và khuyến khích nhân viên sử dụng thời gian nghỉ phép.',
                'expected_impact': 'Cải thiện 25-30% tình trạng sức khỏe tinh thần'
            })

        if 'Social_Isolation' in str(sorted_features[:3]):
            recommendations.append({
                'priority': 'Cao',
                'area': 'Tương tác xã hội',
                'recommendation': 'Tổ chức các hoạt động team building định kỳ và tạo không gian làm việc chung thân thiện.',
                'expected_impact': 'Giảm 20-25% mức độ cô lập xã hội'
            })

        if 'Hours_Per_Week' in str(sorted_features[:3]):
            recommendations.append({
                'priority': 'Trung bình',
                'area': 'Quản lý thời gian',
                'recommendation': 'Giám sát và kiểm soát số giờ làm việc, tránh tình trạng làm việc quá tải thường xuyên.',
                'expected_impact': 'Cải thiện 15-20% tình trạng burnout'
            })

    return recommendations


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/initial_load')
def api_initial_load():
    """API endpoint for automatic initial data loading"""
    try:
        success, message = load_data()
        if success:
            # Get all initial data
            basic_stats = {
                'total_records': int(len(df)),
                'total_columns': int(len(df.columns)),
                'missing_values': int(df.isnull().sum().sum()),
                'data_types': {str(k): int(v) for k, v in df.dtypes.value_counts().to_dict().items()},
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB"
            }

            # Convert sample data
            sample_data = df.head().copy()
            for col in sample_data.columns:
                sample_data[col] = sample_data[col].astype(str)
            sample_data = sample_data.fillna('N/A')

            # Create initial visualizations
            plots = create_advanced_visualizations()

            response_data = {
                'status': 'success',
                'message': message,
                'basic_stats': basic_stats,
                'columns': list(df.columns),
                'sample_data': sample_data.to_dict('records'),
                'initial_plots': plots
            }

            return jsonify(response_data)
        else:
            return jsonify({'status': 'error', 'message': message})

    except Exception as e:
        print(f"Error in api_initial_load: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Lỗi server: {str(e)}'})


@app.route('/api/advanced_visualizations')
def api_advanced_visualizations():
    if df is None:
        return jsonify({'status': 'error', 'message': 'Chưa tải dữ liệu'})

    try:
        plots = create_advanced_visualizations()
        return jsonify({'status': 'success', 'plots': plots})
    except Exception as e:
        print(f"Error in advanced_visualizations: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Lỗi tạo biểu đồ: {str(e)}'})


@app.route('/api/advanced_analysis')
def api_advanced_analysis():
    if df is None:
        return jsonify({'status': 'error', 'message': 'Chưa tải dữ liệu'})

    try:
        analyzer = DataAnalyzer(df)

        # Cache analysis results
        if 'advanced_analysis' not in analysis_cache:
            analysis_cache['advanced_analysis'] = analyzer.advanced_feature_analysis()

        result = analysis_cache['advanced_analysis']

        if result is None:
            return jsonify({'status': 'error', 'message': 'Không thể thực hiện phân tích nâng cao'})

        # Format results for frontend
        formatted_result = {
            'status': 'success',
            'models_performance': {
                name: {
                    'accuracy': data['accuracy'],
                    'cv_mean': data['cv_mean'],
                    'cv_std': data['cv_std']
                } for name, data in result['models_performance'].items()
            },
            'feature_importance': result['feature_importance'],
            'best_model': result['best_model'],
            'recommendations': generate_recommendations(result)
        }

        return jsonify(formatted_result)

    except Exception as e:
        print(f"Error in advanced_analysis: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Lỗi phân tích: {str(e)}'})


@app.route('/api/clustering_analysis')
def api_clustering_analysis():
    if df is None:
        return jsonify({'status': 'error', 'message': 'Chưa tải dữ liệu'})

    try:
        analyzer = DataAnalyzer(df)

        if 'clustering' not in analysis_cache:
            analysis_cache['clustering'] = analyzer.clustering_analysis()

        result = analysis_cache['clustering']

        if result is None:
            return jsonify({'status': 'error', 'message': 'Không thể thực hiện phân tích clustering'})

        # Create clustering visualization
        cluster_viz = create_clustering_visualization(result)

        return jsonify({
            'status': 'success',
            'clustering_results': {
                'optimal_k': result['optimal_k'],
                'cluster_summary': result['cluster_summary'],
                'silhouette_scores': result['silhouette_scores']
            },
            'visualization': cluster_viz
        })

    except Exception as e:
        print(f"Error in clustering_analysis: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Lỗi phân tích clustering: {str(e)}'})


@app.route('/api/statistical_tests')
def api_statistical_tests():
    if df is None:
        return jsonify({'status': 'error', 'message': 'Chưa tải dữ liệu'})

    try:
        tests_results = perform_statistical_tests()
        return jsonify({'status': 'success', 'tests': tests_results})
    except Exception as e:
        print(f"Error in statistical_tests: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Lỗi thử nghiệm thống kê: {str(e)}'})


# Static file serving
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


# Auto-load data on server start
def initialize_data():
    """Automatically load data when server starts"""
    success, message = load_data()
    if success:
        print(f"✅ Auto-loaded data on server start: {message}")
    else:
        print(f"⚠️ Could not auto-load data: {message}")


# Enhanced Analytics Module - Thêm vào app.py

import seaborn as sns
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.manifold import TSNE
import networkx as nx


class EnhancedAnalyzer:
    """Enhanced analytics với focus vào insights thay vì ML accuracy"""

    def __init__(self, dataframe):
        self.df = dataframe
        self.insights = {}

    def risk_scoring_analysis(self):
        """Phân tích điểm rủi ro sức khỏe tinh thần"""
        risk_factors = {
            'work_overload': 0,
            'social_isolation': 0,
            'work_life_imbalance': 0,
            'burnout_risk': 0
        }

        # Calculate risk scores
        if 'Hours_Per_Week' in self.df.columns:
            risk_factors['work_overload'] = (self.df['Hours_Per_Week'] > 50).mean() * 100

        if 'Social_Isolation_Score' in self.df.columns:
            risk_factors['social_isolation'] = (self.df['Social_Isolation_Score'] > self.df[
                'Social_Isolation_Score'].median()).mean() * 100

        if 'Work_Life_Balance_Score' in self.df.columns:
            risk_factors['work_life_imbalance'] = (self.df['Work_Life_Balance_Score'] < 5).mean() * 100

        if 'Burnout_Level' in self.df.columns:
            high_burnout = self.df['Burnout_Level'].isin(['High', 'Medium']).mean() * 100
            risk_factors['burnout_risk'] = high_burnout

        # Calculate composite risk index
        composite_risk = np.mean(list(risk_factors.values()))

        return {
            'risk_factors': risk_factors,
            'composite_risk': composite_risk,
            'risk_level': 'Cao' if composite_risk > 60 else 'Trung bình' if composite_risk > 40 else 'Thấp'
        }

    def demographic_insights(self):
        """Phân tích insights theo nhóm nhân khẩu học"""
        insights = {}

        # Age group analysis
        if 'Age' in self.df.columns and 'Mental_Health_Status' in self.df.columns:
            age_groups = pd.cut(self.df['Age'], bins=[0, 25, 35, 45, 55, 100],
                                labels=['Gen Z', 'Millennials', 'Gen X', 'Boomers', 'Senior'])

            mental_by_age = pd.crosstab(age_groups, self.df['Mental_Health_Status'], normalize='index') * 100
            insights['age_risk'] = mental_by_age.to_dict('index')

        # Gender analysis
        if 'Gender' in self.df.columns and 'Burnout_Level' in self.df.columns:
            burnout_by_gender = pd.crosstab(self.df['Gender'], self.df['Burnout_Level'], normalize='index') * 100
            insights['gender_burnout'] = burnout_by_gender.to_dict('index')

        # Industry analysis
        if 'Industry' in self.df.columns and 'Mental_Health_Status' in self.df.columns:
            industry_risk = self.df.groupby('Industry')['Mental_Health_Status'].apply(
                lambda x: (x != 'Good').mean() * 100
            ).sort_values(ascending=False)
            insights['high_risk_industries'] = industry_risk.head(5).to_dict()

        return insights

    def predictive_indicators(self):
        """Xác định các chỉ báo dự đoán quan trọng nhất"""
        if 'Mental_Health_Status' not in self.df.columns:
            return None

        # Prepare data
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        X = self.df[numeric_cols].fillna(0)

        # Encode target
        y = (self.df['Mental_Health_Status'] != 'Good').astype(int)

        # Mutual Information
        mi_scores = mutual_info_classif(X, y)
        mi_importance = pd.DataFrame({
            'feature': numeric_cols,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)

        # Statistical correlation
        correlations = {}
        for col in numeric_cols:
            corr, p_value = stats.pointbiserialr(y, X[col])
            correlations[col] = {
                'correlation': abs(corr),
                'p_value': p_value,
                'significant': p_value < 0.05
            }

        return {
            'mutual_information': mi_importance.head(10).to_dict('records'),
            'correlations': correlations,
            'top_predictors': mi_importance.head(5)['feature'].tolist()
        }

    def cohort_analysis(self):
        """Phân tích theo cohort để tìm patterns"""
        cohorts = {}

        # Work arrangement cohorts
        if 'Work_Arrangement' in self.df.columns:
            for arrangement in self.df['Work_Arrangement'].unique():
                cohort = self.df[self.df['Work_Arrangement'] == arrangement]
                cohorts[arrangement] = {
                    'size': len(cohort),
                    'avg_isolation': cohort[
                        'Social_Isolation_Score'].mean() if 'Social_Isolation_Score' in cohort.columns else None,
                    'avg_wlb': cohort[
                        'Work_Life_Balance_Score'].mean() if 'Work_Life_Balance_Score' in cohort.columns else None,
                    'mental_health_distribution': cohort['Mental_Health_Status'].value_counts(
                        normalize=True).to_dict() if 'Mental_Health_Status' in cohort.columns else None
                }

        return cohorts

    def anomaly_detection(self):
        """Phát hiện các outliers và anomalies"""
        from sklearn.ensemble import IsolationForest

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None

        X = self.df[numeric_cols].fillna(0)

        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(X)

        # Get anomaly indices
        anomaly_indices = np.where(anomalies == -1)[0]

        # Analyze anomalies
        anomaly_analysis = {
            'total_anomalies': len(anomaly_indices),
            'percentage': (len(anomaly_indices) / len(self.df)) * 100,
            'anomaly_characteristics': {}
        }

        if len(anomaly_indices) > 0:
            anomaly_df = self.df.iloc[anomaly_indices]
            normal_df = self.df.iloc[np.where(anomalies == 1)[0]]

            for col in numeric_cols:
                anomaly_analysis['anomaly_characteristics'][col] = {
                    'anomaly_mean': anomaly_df[col].mean(),
                    'normal_mean': normal_df[col].mean(),
                    'difference': anomaly_df[col].mean() - normal_df[col].mean()
                }

        return anomaly_analysis


def create_advanced_visualizations(df, analyzer):
    """Enhanced visualizations focusing on insights"""
    plots = {}

    try:
        # 1. Risk Dashboard với Gauge Charts
        risk_analysis = analyzer.risk_scoring_analysis()

        fig_risk = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=('Quá tải công việc', 'Cô lập xã hội',
                            'Mất cân bằng công việc-cuộc sống', 'Rủi ro Burnout')
        )

        risk_items = list(risk_analysis['risk_factors'].items())
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for (name, value), (row, col) in zip(risk_items, positions):
            fig_risk.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': name.replace('_', ' ').title()},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if value > 60 else "orange" if value > 40 else "green"},
                        'steps': [
                            {'range': [0, 40], 'color': "lightgray"},
                            {'range': [40, 60], 'color': "gray"},
                            {'range': [60, 100], 'color': "lightgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 60
                        }
                    }
                ),
                row=row, col=col
            )

        fig_risk.update_layout(height=600, title_text="Risk Assessment Dashboard")
        plots['risk_dashboard'] = json.dumps(fig_risk, cls=plotly.utils.PlotlyJSONEncoder)

        # 2. Demographic Heatmap
        insights = analyzer.demographic_insights()

        if 'age_risk' in insights:
            age_data = pd.DataFrame(insights['age_risk']).T
            fig_heatmap = px.imshow(age_data,
                                    labels=dict(x="Mental Health Status", y="Age Group", color="Percentage"),
                                    title="Mental Health Status by Age Group (%)",
                                    color_continuous_scale="RdYlGn_r")
            plots['demographic_heatmap'] = json.dumps(fig_heatmap, cls=plotly.utils.PlotlyJSONEncoder)

        # 3. Predictive Indicators Waterfall
        predictors = analyzer.predictive_indicators()
        if predictors:
            mi_data = predictors['mutual_information'][:10]

            fig_waterfall = go.Figure(go.Waterfall(
                name="Feature Importance",
                orientation="v",
                measure=["relative"] * len(mi_data),
                x=[item['feature'] for item in mi_data],
                y=[item['importance'] for item in mi_data],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))

            fig_waterfall.update_layout(
                title="Feature Importance Analysis (Mutual Information)",
                showlegend=False,
                height=400
            )
            plots['feature_waterfall'] = json.dumps(fig_waterfall, cls=plotly.utils.PlotlyJSONEncoder)

        # 4. Cohort Comparison Radar Chart
        cohorts = analyzer.cohort_analysis()
        if cohorts and len(cohorts) > 1:
            categories = ['Isolation Score', 'Work-Life Balance', 'Team Size']

            fig_radar = go.Figure()

            for name, data in cohorts.items():
                if data['avg_isolation'] is not None and data['avg_wlb'] is not None:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[data['avg_isolation'], data['avg_wlb'], data['size'] / 10],
                        theta=categories,
                        fill='toself',
                        name=name
                    ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )),
                showlegend=True,
                title="Work Arrangement Comparison"
            )
            plots['cohort_radar'] = json.dumps(fig_radar, cls=plotly.utils.PlotlyJSONEncoder)

        # 5. Anomaly Detection Scatter
        anomalies = analyzer.anomaly_detection()
        if anomalies and 'Age' in df.columns and 'Hours_Per_Week' in df.columns:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            X = df[['Age', 'Hours_Per_Week']].fillna(0)
            predictions = iso_forest.fit_predict(X)

            fig_anomaly = px.scatter(
                x=df['Age'],
                y=df['Hours_Per_Week'],
                color=predictions,
                color_discrete_map={1: 'blue', -1: 'red'},
                labels={'color': 'Anomaly'},
                title='Anomaly Detection: Age vs Hours Per Week'
            )
            plots['anomaly_scatter'] = json.dumps(fig_anomaly, cls=plotly.utils.PlotlyJSONEncoder)

    except Exception as e:
        print(f"Error in advanced visualizations: {str(e)}")

    return plots


# API Routes to add to app.py
@app.route('/api/enhanced_analytics')
def api_enhanced_analytics():
    if df is None:
        return jsonify({'status': 'error', 'message': 'Chưa tải dữ liệu'})

    try:
        analyzer = EnhancedAnalyzer(df)

        # Get all enhanced analyses
        risk_analysis = analyzer.risk_scoring_analysis()
        demographic_insights = analyzer.demographic_insights()
        predictive_indicators = analyzer.predictive_indicators()
        cohort_analysis = analyzer.cohort_analysis()
        anomaly_detection = analyzer.anomaly_detection()

        # Create enhanced visualizations
        plots = create_advanced_visualizations(df, analyzer)

        return jsonify({
            'status': 'success',
            'risk_analysis': risk_analysis,
            'demographic_insights': demographic_insights,
            'predictive_indicators': predictive_indicators,
            'cohort_analysis': cohort_analysis,
            'anomaly_detection': anomaly_detection,
            'visualizations': plots
        })

    except Exception as e:
        print(f"Error in enhanced analytics: {str(e)}")
        return jsonify({'status': 'error', 'message': f'Lỗi phân tích: {str(e)}'})

if __name__ == '__main__':
    # Initialize data when starting the app
    initialize_data()
    app.run(debug=True, host='0.0.0.0', port=5000)

