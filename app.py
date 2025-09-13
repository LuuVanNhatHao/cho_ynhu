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
from scipy.stats import chi2_contingency, pearsonr
from datetime import datetime
import warnings

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
        """Tiền xử lý dữ liệu nâng cao"""
        self.processed_df = self.df.copy()

        # Handle missing values
        numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.processed_df.select_dtypes(include=['object']).columns

        # Fill missing values
        for col in numeric_cols:
            self.processed_df[col].fillna(self.processed_df[col].median(), inplace=True)

        for col in categorical_cols:
            self.processed_df[col].fillna(self.processed_df[col].mode()[0], inplace=True)

        # Encode categorical variables
        for col in categorical_cols:
            if col != 'Mental_Health_Status':  # Don't encode target variable yet
                le = LabelEncoder()
                self.processed_df[f'{col}_encoded'] = le.fit_transform(self.processed_df[col].astype(str))
                self.encoders[col] = le

        return self.processed_df

    def advanced_feature_analysis(self):
        """Phân tích feature nâng cao với nhiều thuật toán"""
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
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance_all[name] = dict(zip(feature_cols, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance_all[name] = dict(zip(feature_cols, np.abs(model.coef_[0])))

        return {
            'models_performance': results,
            'feature_importance': feature_importance_all,
            'feature_names': feature_cols,
            'target_encoder': le_target,
            'best_model': max(results.keys(), key=lambda k: results[k]['accuracy'])
        }

    def clustering_analysis(self):
        """Phân tích clustering để tìm nhóm nhân viên"""
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
        K_range = range(2, 8)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_cluster_scaled)
            inertias.append(kmeans.inertia_)

            from sklearn.metrics import silhouette_score
            score = silhouette_score(X_cluster_scaled, kmeans.labels_)
            silhouette_scores.append(score)

        # Choose optimal k (highest silhouette score)
        optimal_k = K_range[np.argmax(silhouette_scores)]

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
                'size': len(cluster_data),
                'avg_age': cluster_data['Age'].mean() if 'Age' in cluster_data.columns else 0,
                'avg_hours': cluster_data['Hours_Per_Week'].mean() if 'Hours_Per_Week' in cluster_data.columns else 0,
                'avg_work_life_balance': cluster_data[
                    'Work_Life_Balance_Score'].mean() if 'Work_Life_Balance_Score' in cluster_data.columns else 0,
                'avg_isolation': cluster_data[
                    'Social_Isolation_Score'].mean() if 'Social_Isolation_Score' in cluster_data.columns else 0,
                'mental_health_dist': cluster_data[
                    'Mental_Health_Status'].value_counts().to_dict() if 'Mental_Health_Status' in cluster_data.columns else {}
            }

        return {
            'optimal_k': optimal_k,
            'clusters': clusters,
            'cluster_summary': cluster_summary,
            'silhouette_scores': list(zip(K_range, silhouette_scores)),
            'cluster_data': cluster_df
        }

    def correlation_network_analysis(self):
        """Phân tích mạng tương quan nâng cao"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()

        # Find strong correlations (>0.5 or <-0.5)
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val,
                        'strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate'
                    })

        return {
            'correlation_matrix': corr_matrix,
            'strong_correlations': strong_correlations
        }


def load_data():
    """Load and validate data"""
    global df
    try:
        df = pd.read_csv('data.csv')
        # Basic data validation
        if df.empty:
            return False, "File CSV trống"
        if len(df.columns) < 5:
            return False, "File CSV không đủ cột dữ liệu"
        return True, f"Đã tải thành công {len(df)} dòng dữ liệu với {len(df.columns)} cột"
    except FileNotFoundError:
        return False, "Không tìm thấy file data.csv"
    except Exception as e:
        return False, f"Lỗi khi đọc file: {str(e)}"


def create_advanced_visualizations():
    """Tạo các biểu đồ nâng cao và tương tác"""
    if df is None:
        return {}

    plots = {}

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
                go.Bar(x=data['Industry'], y=data['count'], name=f'Burnout: {burnout}'),
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
    if all(col in df.columns for col in ['Age', 'Hours_Per_Week', 'Work_Life_Balance_Score', 'Mental_Health_Status']):
        fig_3d = px.scatter_3d(df,
                               x='Age',
                               y='Hours_Per_Week',
                               z='Work_Life_Balance_Score',
                               color='Mental_Health_Status',
                               title='Phân tích 3D: Tuổi - Giờ làm việc - Work-Life Balance',
                               labels={'Age': 'Tuổi', 'Hours_Per_Week': 'Giờ/tuần',
                                       'Work_Life_Balance_Score': 'Điểm WLB'})
        plots['scatter_3d'] = json.dumps(fig_3d, cls=plotly.utils.PlotlyJSONEncoder)

    # 3. Heatmap chi tiết
    if len(df.select_dtypes(include=[np.number]).columns) > 3:
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()

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

    # 5. Violin plots comparison
    if all(col in df.columns for col in ['Work_Arrangement', 'Social_Isolation_Score']):
        fig_violin = px.violin(df,
                               x='Work_Arrangement',
                               y='Social_Isolation_Score',
                               color='Work_Arrangement',
                               title='Mức độ cô lập theo hình thức làm việc')
        plots['violin_arrangement'] = json.dumps(fig_violin, cls=plotly.utils.PlotlyJSONEncoder)

    return plots


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/load_data')
def api_load_data():
    success, message = load_data()
    if success:
        analyzer = DataAnalyzer(df)
        basic_stats = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'data_types': df.dtypes.value_counts().to_dict(),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB"
        }

        return jsonify({
            'status': 'success',
            'message': message,
            'basic_stats': basic_stats,
            'columns': df.columns.tolist(),
            'sample_data': df.head().to_dict('records')
        })
    else:
        return jsonify({'status': 'error', 'message': message})


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
        return jsonify({'status': 'error', 'message': f'Lỗi phân tích clustering: {str(e)}'})


@app.route('/api/advanced_visualizations')
def api_advanced_visualizations():
    if df is None:
        return jsonify({'status': 'error', 'message': 'Chưa tải dữ liệu'})

    try:
        plots = create_advanced_visualizations()
        return jsonify({'status': 'success', 'plots': plots})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Lỗi tạo biểu đồ: {str(e)}'})


@app.route('/api/statistical_tests')
def api_statistical_tests():
    if df is None:
        return jsonify({'status': 'error', 'message': 'Chưa tải dữ liệu'})

    try:
        tests_results = perform_statistical_tests()
        return jsonify({'status': 'success', 'tests': tests_results})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Lỗi thử nghiệm thống kê: {str(e)}'})


def create_clustering_visualization(cluster_result):
    """Tạo biểu đồ clustering"""
    cluster_data = cluster_result['cluster_data']

    # 2D projection using PCA
    from sklearn.decomposition import PCA

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

    return None


def perform_statistical_tests():
    """Thực hiện các kiểm định thống kê"""
    results = {}

    try:
        # Chi-square test for categorical variables
        if all(col in df.columns for col in ['Gender', 'Mental_Health_Status']):
            contingency_table = pd.crosstab(df['Gender'], df['Mental_Health_Status'])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            results['chi_square_gender_mental'] = {
                'test_name': 'Chi-square: Giới tính vs Sức khỏe tinh thần',
                'statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'interpretation': 'Có mối liên hệ có ý nghĩa' if p_value < 0.05 else 'Không có mối liên hệ có ý nghĩa'
            }

        # T-test for work hours vs mental health
        if all(col in df.columns for col in ['Hours_Per_Week', 'Mental_Health_Status']):
            from scipy.stats import ttest_ind

            mental_categories = df['Mental_Health_Status'].unique()
            if len(mental_categories) >= 2:
                group1 = df[df['Mental_Health_Status'] == mental_categories[0]]['Hours_Per_Week'].dropna()
                group2 = df[df['Mental_Health_Status'] == mental_categories[1]]['Hours_Per_Week'].dropna()

                t_stat, p_value = ttest_ind(group1, group2)

                results['t_test_hours_mental'] = {
                    'test_name': f'T-test: Giờ làm việc giữa {mental_categories[0]} và {mental_categories[1]}',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'interpretation': 'Có sự khác biệt có ý nghĩa' if p_value < 0.05 else 'Không có sự khác biệt có ý nghĩa'
                }

        # ANOVA for work-life balance across mental health categories
        if all(col in df.columns for col in ['Work_Life_Balance_Score', 'Mental_Health_Status']):
            from scipy.stats import f_oneway

            groups = [group['Work_Life_Balance_Score'].dropna() for name, group in df.groupby('Mental_Health_Status')]
            if len(groups) >= 2:
                f_stat, p_value = f_oneway(*groups)

                results['anova_wlb_mental'] = {
                    'test_name': 'ANOVA: Work-Life Balance giữa các nhóm sức khỏe tinh thần',
                    'statistic': f_stat,
                    'p_value': p_value,
                    'interpretation': 'Có sự khác biệt có ý nghĩa giữa các nhóm' if p_value < 0.05 else 'Không có sự khác biệt có ý nghĩa'
                }

    except Exception as e:
        results['error'] = f'Lỗi thực hiện kiểm định: {str(e)}'

    return results


def generate_recommendations(analysis_result):
    """Tạo khuyến nghị dựa trên kết quả phân tích"""
    recommendations = []

    if 'feature_importance' in analysis_result:
        best_model = analysis_result['best_model']
        feature_importance = analysis_result['feature_importance'][best_model]

        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        vietnamese_names = {
            'Work_Life_Balance_Score': 'Cân bằng công việc-cuộc sống',
            'Social_Isolation_Score': 'Mức độ cô lập xã hội',
            'Hours_Per_Week': 'Số giờ làm việc/tuần',
            'Age': 'Độ tuổi',
            'Industry_encoded': 'Ngành nghề',
            'Work_Arrangement_encoded': 'Hình thức làm việc',
            'Job_Role_encoded': 'Vai trò công việc',
            'Physical_Health_Issues_encoded': 'Vấn đề sức khỏe thể chất',
            'Salary_Range_encoded': 'Mức lương',
            'Gender_encoded': 'Giới tính',
            'Region_encoded': 'Khu vực'
        }

        # Generate specific recommendations
        top_feature = sorted_features[0][0]
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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)