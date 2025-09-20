"""
WorkWell Analytics - Flask Application
Phân tích Work-Life Balance với EDA, Hypothesis Testing và Clustering
"""

from flask import Flask, render_template, jsonify, request, session
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import pickle

# Import services
from services.data_loader import DataLoader
from services.analytics import AnalyticsEngine
from services.viz import VizEngine
from config import Config

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.datetime64, pd.Timestamp)):
            return str(obj)
        return super(NumpyEncoder, self).default(obj)

def safe_jsonify(data):
    """Create a JSON response with numpy-safe encoding"""
    return app.response_class(
        response=json.dumps(data, cls=NumpyEncoder, ensure_ascii=False),
        status=200,
        mimetype='application/json'
    )

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY

# Global variables để cache data
data_loader = None
analytics_engine = None
viz_engine = None
df_main = None

def init_app():
    """Initialize app với data preload"""
    global data_loader, analytics_engine, viz_engine, df_main

    print("🚀 Initializing WorkWell Analytics...")

    # Load data
    data_loader = DataLoader(Config.DATA_PATH)
    df_main = data_loader.load_and_preprocess()
    print(f"✅ Loaded {len(df_main)} records")

    # Initialize engines
    analytics_engine = AnalyticsEngine(df_main)
    viz_engine = VizEngine()

    # Pre-compute và cache các phân tích nặng
    print("📊 Pre-computing analytics...")
    analytics_engine.precompute_all()

    print("✨ App ready!")
    return True

@app.route('/')
def index():
    """Landing page với overview"""
    stats = {
        'total_records': len(df_main),
        'regions': df_main['Region'].nunique(),
        'industries': df_main['Industry'].nunique(),
        'avg_hours': df_main['Hours_Per_Week'].mean(),
        'burnout_rate': (df_main['Burnout_Level'].isin(['High', 'Very High']).sum() / len(df_main) * 100)
    }
    return render_template('index.html', stats=stats)

@app.route('/eda')
def eda_page():
    """EDA Dashboard"""
    return render_template('eda.html')

@app.route('/api/eda/<chart_type>')
def get_eda_chart(chart_type):
    """API endpoint cho các chart EDA"""
    try:
        if chart_type == 'demographics':
            fig = viz_engine.plot_demographics(df_main)
        elif chart_type == 'work_arrangement':
            fig = viz_engine.plot_work_arrangement(df_main)
        elif chart_type == 'burnout_distribution':
            fig = viz_engine.plot_burnout_by_factors(df_main)
        elif chart_type == 'correlation_matrix':
            fig = viz_engine.plot_correlation_matrix(df_main)
        elif chart_type == 'health_issues':
            fig = viz_engine.plot_health_issues(df_main)
        elif chart_type == 'industry_comparison':
            fig = viz_engine.plot_industry_comparison(df_main)
        else:
            return jsonify({'error': 'Invalid chart type'}), 400

        # Convert Plotly figure to dictionary format
        return safe_jsonify({
            'data': fig.to_dict()['data'],
            'layout': fig.to_dict()['layout']
        })
    except Exception as e:
        print(f"Error in get_eda_chart: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/hypothesis')
def hypothesis_page():
    """Hypothesis Testing Page"""
    # Get precomputed results
    results = analytics_engine.get_hypothesis_results()
    return render_template('hypothesis.html', results=results)

@app.route('/api/hypothesis/<test_type>')
def run_hypothesis_test(test_type):
    """API cho hypothesis testing"""
    try:
        if test_type == 'summary':
            # Return summary of all tests
            return safe_jsonify({'tests': analytics_engine.get_hypothesis_summary()})
        elif test_type == 'remote_burnout':
            result = analytics_engine.test_remote_vs_burnout()
        elif test_type == 'gender_balance':
            result = analytics_engine.test_gender_balance_score()
        elif test_type == 'age_mental_health':
            result = analytics_engine.test_age_mental_health()
        elif test_type == 'industry_hours':
            result = analytics_engine.test_industry_hours()
        elif test_type == 'salary_isolation':
            result = analytics_engine.test_salary_isolation()
        else:
            return jsonify({'error': 'Invalid test type'}), 400

        # Generate visualization
        fig = viz_engine.plot_hypothesis_result(result, df_main)

        return safe_jsonify({
            'result': result,
            'chart': {
                'data': fig.to_dict()['data'],
                'layout': fig.to_dict()['layout']
            } if fig else None
        })
    except Exception as e:
        print(f"Error in run_hypothesis_test: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/clusters')
def clusters_page():
    """Clustering & Persona Analysis"""
    return render_template('clusters.html')

@app.route('/api/clusters/compute', methods=['POST'])
def compute_clusters():
    """Compute hoặc get cached clusters"""
    try:
        params = request.json or {}
        n_clusters = params.get('n_clusters', 5)
        method = params.get('method', 'kmeans')

        # Get clustering results
        cluster_results = analytics_engine.perform_clustering(
            n_clusters=n_clusters,
            method=method
        )

        # Generate visualizations
        charts = {}

        # UMAP plot
        umap_fig = viz_engine.plot_cluster_umap(cluster_results)
        charts['umap_plot'] = {
            'data': umap_fig.to_dict()['data'],
            'layout': umap_fig.to_dict()['layout']
        }

        # Profile plot
        profile_fig = viz_engine.plot_cluster_profile(cluster_results)
        charts['profile'] = {
            'data': profile_fig.to_dict()['data'],
            'layout': profile_fig.to_dict()['layout']
        }

        # Distribution plot
        dist_fig = viz_engine.plot_cluster_distribution(cluster_results)
        charts['distribution'] = {
            'data': dist_fig.to_dict()['data'],
            'layout': dist_fig.to_dict()['layout']
        }

        # Generate personas
        personas = analytics_engine.generate_personas(cluster_results)

        return safe_jsonify({
            'success': True,
            'n_clusters': n_clusters,
            'silhouette_score': cluster_results['silhouette_score'],
            'charts': charts,
            'personas': personas
        })
    except Exception as e:
        print(f"Error in compute_clusters: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clusters/drivers', methods=['POST'])
def get_cluster_drivers():
    """Get feature importance for clusters"""
    try:
        cluster_id = request.json.get('cluster_id', 0)
        drivers = analytics_engine.get_cluster_drivers(cluster_id)

        # Visualize drivers
        fig = viz_engine.plot_feature_importance(drivers)

        return safe_jsonify({
            'drivers': drivers,
            'chart': fig.to_dict()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<export_type>')
def export_data(export_type):
    """Export functionality"""
    try:
        if export_type == 'eda_report':
            report = analytics_engine.generate_eda_report()
            return safe_jsonify(report)
        elif export_type == 'hypothesis_summary':
            summary = analytics_engine.get_hypothesis_summary()
            return safe_jsonify(summary)
        elif export_type == 'personas':
            personas = analytics_engine.export_personas()
            return safe_jsonify(personas)
        else:
            return jsonify({'error': 'Invalid export type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/filter', methods=['POST'])
def apply_filters():
    """Apply filters to data"""
    try:
        filters = request.json
        df_filtered = data_loader.apply_filters(df_main, filters)

        # Update analytics engine with filtered data
        analytics_engine.update_data(df_filtered)

        return safe_jsonify({
            'success': True,
            'records_count': len(df_filtered),
            'filters_applied': filters
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Initialize app với data preload
    if init_app():
        # Run Flask app
        app.run(
            debug=Config.DEBUG,
            host=Config.HOST,
            port=Config.PORT,
            use_reloader=False  # Tắt reloader để tránh load data 2 lần
        )
    else:
        print("❌ Failed to initialize app")