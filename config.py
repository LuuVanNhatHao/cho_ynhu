"""
Configuration for WorkWell Analytics App
"""

import os
from datetime import timedelta


class Config:
    """Base configuration"""

    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'workwell-secret-key-2024'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    HOST = '0.0.0.0'
    PORT = 5000

    # Data paths
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATA_PATH = os.path.join(DATA_DIR, 'data.csv')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    CACHE_DIR = os.path.join(BASE_DIR, 'cache')

    # Create directories if not exist
    for dir_path in [DATA_DIR, MODELS_DIR, CACHE_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # Analytics settings
    RANDOM_STATE = 42
    TEST_SIZE = 0.2

    # Clustering defaults
    DEFAULT_N_CLUSTERS = 5
    CLUSTERING_METHODS = ['kmeans', 'gaussian_mixture', 'hierarchical']
    UMAP_N_NEIGHBORS = 15
    UMAP_MIN_DIST = 0.1

    # Hypothesis testing
    SIGNIFICANCE_LEVEL = 0.05
    EFFECT_SIZE_THRESHOLDS = {
        'small': 0.2,
        'medium': 0.5,
        'large': 0.8
    }

    # Visualization settings
    PLOTLY_THEME = 'plotly_white'
    COLOR_PALETTE = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    CHART_HEIGHT = 500
    CHART_WIDTH = 900

    # Cache settings
    CACHE_TIMEOUT = timedelta(hours=1)
    ENABLE_CACHE = True

    # Export settings
    EXPORT_FORMATS = ['json', 'csv', 'xlsx', 'html']
    MAX_EXPORT_ROWS = 10000

    # Session settings
    SESSION_TYPE = 'filesystem'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)

    # Performance settings
    CHUNK_SIZE = 1000  # For batch processing
    MAX_WORKERS = 4  # For parallel processing

    # Feature engineering settings
    CATEGORICAL_FEATURES = [
        'Gender', 'Region', 'Industry', 'Job_Role',
        'Work_Arrangement', 'Mental_Health_Status',
        'Burnout_Level', 'Physical_Health_Issues', 'Salary_Range'
    ]

    NUMERICAL_FEATURES = [
        'Age', 'Hours_Per_Week', 'Work_Life_Balance_Score',
        'Social_Isolation_Score'
    ]

    ORDINAL_MAPPINGS = {
        'Burnout_Level': {
            'Low': 1,
            'Medium': 2,
            'High': 3,
            'Very High': 4
        },
        'Mental_Health_Status': {
            'Good': 3,
            'Average': 2,
            'Poor': 1
        },
        'Salary_Range': {
            '<50k': 1,
            '50k-75k': 2,
            '75k-100k': 3,
            '100k-150k': 4,
            '>150k': 5
        }
    }

    # Persona generation settings
    PERSONA_NAMES = {
        0: "The Balanced Professional",
        1: "The Overworked Executive",
        2: "The Remote Worker",
        3: "The Junior Struggler",
        4: "The Senior Optimizer"
    }

    # API rate limiting (if needed)
    RATE_LIMIT = '100 per hour'

    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = os.path.join(BASE_DIR, 'workwell.log')


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

    # Override with environment variables in production
    SECRET_KEY = os.environ.get('SECRET_KEY')

    # Security headers
    SECURITY_HEADERS = {
        'X-Frame-Options': 'SAMEORIGIN',
        'X-Content-Type-Options': 'nosniff',
        'X-XSS-Protection': '1; mode=block'
    }


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True

    # Use in-memory database for tests
    DATA_PATH = ':memory:'


# Config selector
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    """Get config based on environment"""
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])