"""
Quick fix script for WorkWell Analytics issues
Run this to diagnose and fix common problems
"""

import os
import sys
import json
from pathlib import Path


def check_requirements():
    """Check if all required packages are installed"""
    print("Checking requirements...")
    missing = []

    required = [
        'flask', 'pandas', 'numpy', 'scipy', 'scikit-learn',
        'plotly', 'umap', 'statsmodels'
    ]

    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)

    if missing:
        print(f"\nPlease install missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True


def check_structure():
    """Check if directory structure is correct"""
    print("\nChecking directory structure...")

    dirs_needed = ['data', 'models', 'cache', 'services', 'templates', 'static/css', 'static/js']
    for dir_path in dirs_needed:
        path = Path(dir_path)
        if not path.exists():
            print(f"Creating {dir_path}/")
            path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✓ {dir_path}/")


def check_files():
    """Check if all required files exist"""
    print("\nChecking files...")

    files_needed = {
        'app.py': 'Main application',
        'config.py': 'Configuration',
        'services/data_loader.py': 'Data loader service',
        'services/analytics.py': 'Analytics engine',
        'services/viz.py': 'Visualization engine',
        'templates/base.html': 'Base template',
        'templates/index.html': 'Index page',
        'templates/eda.html': 'EDA page',
        'templates/hypothesis.html': 'Hypothesis page',
        'templates/clusters.html': 'Clusters page',
        'templates/404.html': '404 error page',
        'templates/500.html': '500 error page',
        'static/js/eda.js': 'EDA JavaScript',
        'static/js/hypothesis.js': 'Hypothesis JavaScript',
        'static/js/clusters.js': 'Clusters JavaScript',
        'data/data.csv': 'Data file'
    }

    missing_files = []
    for file_path, description in files_needed.items():
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING ({description})")
            missing_files.append(file_path)

    return missing_files


def create_missing_templates():
    """Create missing template files"""
    print("\nCreating missing templates...")

    # Create 404.html if missing
    if not Path('templates/404.html').exists():
        with open('templates/404.html', 'w', encoding='utf-8') as f:
            f.write('''{% extends "base.html" %}
{% block title %}404 - Page Not Found{% endblock %}
{% block content %}
<div class="container text-center py-5">
    <h1 class="display-1 text-danger">404</h1>
    <h2>Page Not Found</h2>
    <a href="/" class="btn btn-primary mt-3">Go to Dashboard</a>
</div>
{% endblock %}''')
        print("Created templates/404.html")

    # Create 500.html if missing
    if not Path('templates/500.html').exists():
        with open('templates/500.html', 'w', encoding='utf-8') as f:
            f.write('''{% extends "base.html" %}
{% block title %}500 - Server Error{% endblock %}
{% block content %}
<div class="container text-center py-5">
    <h1 class="display-1 text-warning">500</h1>
    <h2>Server Error</h2>
    <a href="/" class="btn btn-primary mt-3">Go to Dashboard</a>
</div>
{% endblock %}''')
        print("Created templates/500.html")


def create_services_init():
    """Create __init__.py files for services"""
    if not Path('services/__init__.py').exists():
        Path('services/__init__.py').touch()
        print("Created services/__init__.py")


def test_data_loading():
    """Test if data can be loaded"""
    print("\nTesting data loading...")
    try:
        import pandas as pd
        df = pd.read_csv('data/data.csv')
        print(f"✓ Data loaded: {len(df)} rows, {len(df.columns)} columns")

        # Check for required columns
        required_cols = [
            'Age', 'Gender', 'Hours_Per_Week', 'Work_Life_Balance_Score',
            'Burnout_Level', 'Mental_Health_Status'
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠ Missing columns: {missing_cols}")
        else:
            print("✓ All required columns present")

    except FileNotFoundError:
        print("✗ data/data.csv not found!")
        print("Please place your data.csv file in the data/ directory")
    except Exception as e:
        print(f"✗ Error loading data: {e}")


def create_favicon_route_fix():
    """Add favicon handling to avoid 404 errors"""
    print("\nAdding favicon handler...")

    fix_code = '''
# Add this to your app.py after other imports
@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content
'''
    print("Add the following code to app.py to handle favicon requests:")
    print(fix_code)


def main():
    print("=" * 50)
    print("WorkWell Analytics - Issue Fixer")
    print("=" * 50)

    # Check requirements
    if not check_requirements():
        print("\n⚠ Please install missing requirements first")
        return

    # Check and create directory structure
    check_structure()

    # Check files
    missing_files = check_files()

    # Create missing templates
    create_missing_templates()

    # Create services __init__.py
    create_services_init()

    # Test data loading
    test_data_loading()

    # Suggest favicon fix
    create_favicon_route_fix()

    print("\n" + "=" * 50)
    if missing_files:
        critical_missing = [f for f in missing_files if not f.endswith('.html') and f != 'data/data.csv']
        if critical_missing:
            print("⚠ CRITICAL FILES MISSING:")
            for f in critical_missing:
                print(f"  - {f}")
            print("\nPlease ensure all Python files are in place")
        else:
            print("✓ All critical files present")
            print("✓ Templates have been created")
            print("\nYou should be able to run the app now:")
            print("  python app.py")
    else:
        print("✓ All files present!")
        print("✓ App should be ready to run:")
        print("  python app.py")


if __name__ == "__main__":
    main()