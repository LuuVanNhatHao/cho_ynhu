// Advanced Analytics Platform - Main JavaScript

const API_BASE = 'http://localhost:5000/api';

// Application State
const AppState = {
    dataLoaded: false,
    schema: null,
    currentSection: 'overview',
    clusteringResults: null,
    statisticalResults: null,
    selectedNumericFeatures: [],
    selectedCategoricalFeatures: []
};

// Utility Functions
const Utils = {
    showLoading: (show = true) => {
        const overlay = document.getElementById('loadingOverlay');
        if (show) {
            overlay.classList.add('show');
        } else {
            overlay.classList.remove('show');
        }
    },

    showAlert: (message, type = 'info', duration = 5000) => {
        const alertContainer = document.getElementById('alertContainer');
        const alertId = `alert_${Date.now()}`;

        const alertHtml = `
            <div id="${alertId}" class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;

        alertContainer.insertAdjacentHTML('beforeend', alertHtml);

        if (duration > 0) {
            setTimeout(() => {
                const alert = document.getElementById(alertId);
                if (alert) {
                    alert.remove();
                }
            }, duration);
        }
    },

    formatNumber: (num) => {
        if (typeof num !== 'number') return '-';
        if (Number.isInteger(num)) return num.toLocaleString();
        return num.toFixed(4);
    },

    downloadFile: (content, filename, type = 'text/plain') => {
        const blob = new Blob([content], { type });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
};

// API Service
const API = {
    async getSchema() {
        try {
            const response = await fetch(`${API_BASE}/schema`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching schema:', error);
            throw error;
        }
    },

    async runClustering(params) {
        try {
            const response = await fetch(`${API_BASE}/cluster`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            return await response.json();
        } catch (error) {
            console.error('Error running clustering:', error);
            throw error;
        }
    },

    async sweepClustering(params) {
        try {
            const response = await fetch(`${API_BASE}/cluster/sweep`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            return await response.json();
        } catch (error) {
            console.error('Error sweeping clusters:', error);
            throw error;
        }
    },

    async runStabilityAnalysis(params) {
        try {
            const response = await fetch(`${API_BASE}/cluster/stability`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            return await response.json();
        } catch (error) {
            console.error('Error running stability analysis:', error);
            throw error;
        }
    },

    async runStatistics(params) {
        try {
            const response = await fetch(`${API_BASE}/statistics`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            return await response.json();
        } catch (error) {
            console.error('Error running statistics:', error);
            throw error;
        }
    },

    async exportResults(params) {
        try {
            const response = await fetch(`${API_BASE}/export`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            return await response.json();
        } catch (error) {
            console.error('Error exporting results:', error);
            throw error;
        }
    }
};

// UI Controllers
const UIController = {
    initNavigation() {
        // Sidebar toggle
        document.getElementById('sidebarToggle').addEventListener('click', () => {
            const sidebar = document.getElementById('sidebar');
            const mainContent = document.getElementById('mainContent');

            sidebar.classList.toggle('collapsed');
            mainContent.classList.toggle('expanded');
        });

        // Navigation items
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const section = e.currentTarget.dataset.section;
                this.switchSection(section);
            });
        });
    },

    switchSection(section) {
        // Update nav items
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-section="${section}"]`).classList.add('active');

        // Update sections
        document.querySelectorAll('.content-section').forEach(sec => {
            sec.classList.remove('active');
        });
        document.getElementById(`${section}-section`).classList.add('active');

        AppState.currentSection = section;
    },

    updateDataStatus(loaded) {
        const status = document.getElementById('dataStatus');
        if (loaded) {
            status.textContent = 'Data Loaded';
            status.className = 'badge bg-success';
        } else {
            status.textContent = 'Not Loaded';
            status.className = 'badge bg-secondary';
        }
    },

    displaySchema(schema) {
        // Update stats
        document.getElementById('totalRows').textContent = schema.n_rows.toLocaleString();
        document.getElementById('totalCols').textContent = schema.n_cols;
        document.getElementById('numericCols').textContent = schema.numeric_cols.length;
        document.getElementById('categoricalCols').textContent = schema.categorical_cols.length;

        // Display numeric columns
        const numericList = document.getElementById('numericColumnsList');
        numericList.innerHTML = schema.numeric_cols.map(col => `
            <div class="column-item">
                <span>${col}</span>
                <span class="badge bg-primary">Numeric</span>
            </div>
        `).join('');

        // Display categorical columns
        const categoricalList = document.getElementById('categoricalColumnsList');
        categoricalList.innerHTML = schema.categorical_cols.map(col => `
            <div class="column-item">
                <span>${col}</span>
                <span class="badge bg-secondary">Categorical</span>
            </div>
        `).join('');

        // Populate feature selectors
        this.populateFeatureSelectors(schema);

        // Display basic statistics
        this.displayBasicStats(schema.basic_stats);

        // Populate target variable selector
        const targetSelect = document.getElementById('targetVariable');
        targetSelect.innerHTML = '<option value="">None</option>' +
            [...schema.numeric_cols, ...schema.categorical_cols].map(col =>
                `<option value="${col}">${col}</option>`
            ).join('');
    },

    populateFeatureSelectors(schema) {
        // Numeric features
        const numericFeatures = document.getElementById('numericFeatures');
        numericFeatures.innerHTML = schema.numeric_cols.map(col => `
            <div class="feature-item" data-feature="${col}" data-type="numeric">
                ${col}
            </div>
        `).join('');

        // Categorical features
        const categoricalFeatures = document.getElementById('categoricalFeatures');
        categoricalFeatures.innerHTML = schema.categorical_cols.map(col => `
            <div class="feature-item" data-feature="${col}" data-type="categorical">
                ${col}
            </div>
        `).join('');

        // Add click handlers
        document.querySelectorAll('.feature-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.currentTarget.classList.toggle('selected');
                this.updateSelectedFeatures();
            });
        });

        // Check available algorithms
        if (schema.optional_algorithms) {
            const algoSelect = document.getElementById('clusterAlgorithm');
            if (schema.optional_algorithms.hdbscan) {
                algoSelect.querySelector('option[value="hdbscan"]').disabled = false;
            }
            if (schema.optional_algorithms.kprototypes) {
                algoSelect.querySelector('option[value="kprototypes"]').disabled = false;
            }
        }
    },

    updateSelectedFeatures() {
        AppState.selectedNumericFeatures = Array.from(
            document.querySelectorAll('#numericFeatures .feature-item.selected')
        ).map(el => el.dataset.feature);

        AppState.selectedCategoricalFeatures = Array.from(
            document.querySelectorAll('#categoricalFeatures .feature-item.selected')
        ).map(el => el.dataset.feature);
    },

    displayBasicStats(stats) {
        if (!stats) return;

        const container = document.getElementById('basicStatsContent');
        let html = '';

        // Numeric statistics
        if (stats.numeric_summary) {
            html += '<h6>Numeric Variables Summary</h6>';
            html += '<div class="table-responsive"><table class="table table-sm">';
            html += '<thead><tr><th>Variable</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Skew</th></tr></thead>';
            html += '<tbody>';

            for (const [col, colStats] of Object.entries(stats.numeric_summary)) {
                html += `
                    <tr>
                        <td><strong>${col}</strong></td>
                        <td>${Utils.formatNumber(colStats.mean)}</td>
                        <td>${Utils.formatNumber(colStats.std)}</td>
                        <td>${Utils.formatNumber(colStats.min)}</td>
                        <td>${Utils.formatNumber(colStats.max)}</td>
                        <td>${Utils.formatNumber(colStats.skew)}</td>
                    </tr>
                `;
            }
            html += '</tbody></table></div>';
        }

        // Categorical statistics
        if (stats.categorical_summary) {
            html += '<h6 class="mt-4">Categorical Variables Summary</h6>';
            html += '<div class="table-responsive"><table class="table table-sm">';
            html += '<thead><tr><th>Variable</th><th>Unique Values</th><th>Entropy</th><th>Top Value</th></tr></thead>';
            html += '<tbody>';

            for (const [col, colStats] of Object.entries(stats.categorical_summary)) {
                const topValue = Object.keys(colStats.top_values)[0] || 'N/A';
                html += `
                    <tr>
                        <td><strong>${col}</strong></td>
                        <td>${colStats.unique_values}</td>
                        <td>${Utils.formatNumber(colStats.entropy)}</td>
                        <td>${topValue}</td>
                    </tr>
                `;
            }
            html += '</tbody></table></div>';
        }

        container.innerHTML = html;
    },

    displayClusteringResults(results) {
        const container = document.getElementById('clusteringResults');

        // Metrics
        let html = '<div class="results-container">';
        html += '<h6>Clustering Metrics</h6>';

        if (results.metrics) {
            html += '<div class="metric-row">';
            html += `<span class="metric-label">Number of Clusters:</span>`;
            html += `<span class="metric-value">${results.metrics.n_clusters || '-'}</span>`;
            html += '</div>';

            if (results.metrics.silhouette !== undefined) {
                const silhouette = results.metrics.silhouette;
                const quality = silhouette > 0.5 ? 'good' : silhouette > 0.25 ? 'warning' : 'bad';
                html += '<div class="metric-row">';
                html += `<span class="metric-label">Silhouette Score:</span>`;
                html += `<span class="metric-value ${quality}">${Utils.formatNumber(silhouette)}</span>`;
                html += '</div>';
            }

            if (results.metrics.davies_bouldin !== undefined) {
                html += '<div class="metric-row">';
                html += `<span class="metric-label">Davies-Bouldin Index:</span>`;
                html += `<span class="metric-value">${Utils.formatNumber(results.metrics.davies_bouldin)}</span>`;
                html += '</div>';
            }

            if (results.metrics.calinski_harabasz !== undefined) {
                html += '<div class="metric-row">';
                html += `<span class="metric-label">Calinski-Harabasz Index:</span>`;
                html += `<span class="metric-value">${Utils.formatNumber(results.metrics.calinski_harabasz)}</span>`;
                html += '</div>';
            }

            if (results.metrics.n_noise !== undefined) {
                html += '<div class="metric-row">';
                html += `<span class="metric-label">Noise Points:</span>`;
                html += `<span class="metric-value">${results.metrics.n_noise} (${(results.metrics.noise_ratio * 100).toFixed(1)}%)</span>`;
                html += '</div>';
            }
        }

        html += '</div>';

        // Cluster Profiles
        if (results.profiles) {
            html += '<div class="mt-4"><h6>Cluster Profiles</h6>';

            for (const [clusterId, profile] of Object.entries(results.profiles)) {
                html += `<div class="cluster-profile">`;
                html += `<h6>Cluster ${clusterId} (n=${profile.size}, ${(profile.share * 100).toFixed(1)}%)</h6>`;

                // Numeric summary
                if (profile.numeric_summary && Object.keys(profile.numeric_summary).length > 0) {
                    html += '<div class="profile-stats">';
                    for (const [col, stats] of Object.entries(profile.numeric_summary)) {
                        const delta = stats.delta_from_global;
                        const effectSize = stats.effect_size;
                        html += `
                            <div class="profile-stat">
                                <div class="profile-stat-label">${col}</div>
                                <div class="profile-stat-value">${Utils.formatNumber(stats.mean)}</div>
                                <small class="${Math.abs(effectSize) > 0.5 ? 'text-primary' : 'text-muted'}">
                                    Δ: ${delta > 0 ? '+' : ''}${Utils.formatNumber(delta)}
                                </small>
                            </div>
                        `;
                    }
                    html += '</div>';
                }

                // Categorical summary
                if (profile.categorical_summary && Object.keys(profile.categorical_summary).length > 0) {
                    html += '<div class="mt-2"><small><strong>Dominant Categories:</strong></small><br>';
                    for (const [col, catStats] of Object.entries(profile.categorical_summary)) {
                        if (catStats.mode) {
                            const lift = catStats.lift[catStats.mode] || 1;
                            html += `<span class="badge bg-secondary me-1">${col}: ${catStats.mode} (lift: ${lift.toFixed(2)})</span>`;
                        }
                    }
                    html += '</div>';
                }

                html += '</div>';
            }

            html += '</div>';
        }

        container.innerHTML = html;

        // Display visualization
        if (results.visualization) {
            const vizContainer = document.getElementById('clusteringVisualization');
            const plotData = JSON.parse(results.visualization);
            Plotly.newPlot(vizContainer, plotData.data, plotData.layout, {responsive: true});
        }
    },

    displayStatisticalResults(results) {
        const container = document.getElementById('statisticalResults');
        let html = '<div class="results-container">';

        // Normality Tests
        if (results.test_results && results.test_results.normality_tests) {
            html += '<h6>Normality Tests</h6>';
            html += '<div class="table-responsive"><table class="table table-sm">';
            html += '<thead><tr><th>Variable</th><th>Test</th><th>Statistic</th><th>P-value</th><th>Normal?</th></tr></thead>';
            html += '<tbody>';

            for (const [col, test] of Object.entries(results.test_results.normality_tests)) {
                html += `
                    <tr>
                        <td>${col}</td>
                        <td>${test.test}</td>
                        <td>${Utils.formatNumber(test.statistic)}</td>
                        <td>${Utils.formatNumber(test.p_value)}</td>
                        <td>${test.is_normal ? 
                            '<span class="badge bg-success">Yes</span>' : 
                            '<span class="badge bg-warning">No</span>'}</td>
                    </tr>
                `;
            }
            html += '</tbody></table></div>';
        }

        // Hypothesis Tests
        if (results.test_results && results.test_results.hypothesis_tests) {
            html += '<h6 class="mt-4">Hypothesis Tests</h6>';
            html += '<div class="table-responsive"><table class="table table-sm">';
            html += '<thead><tr><th>Test</th><th>Type</th><th>Statistic</th><th>P-value</th><th>Significant?</th></tr></thead>';
            html += '<tbody>';

            for (const [testName, test] of Object.entries(results.test_results.hypothesis_tests)) {
                html += `
                    <tr>
                        <td>${testName}</td>
                        <td>${test.test}</td>
                        <td>${Utils.formatNumber(test.statistic)}</td>
                        <td>${Utils.formatNumber(test.p_value)}</td>
                        <td>${test.significant ? 
                            '<span class="badge bg-success">Yes</span>' : 
                            '<span class="badge bg-secondary">No</span>'}</td>
                    </tr>
                `;
            }
            html += '</tbody></table></div>';
        }

        // Outlier Detection
        if (results.outlier_results) {
            html += '<h6 class="mt-4">Outlier Detection</h6>';

            if (results.outlier_results.isolation_forest) {
                const iso = results.outlier_results.isolation_forest;
                html += `<div class="metric-row">
                    <span class="metric-label">Isolation Forest Outliers:</span>
                    <span class="metric-value">${iso.n_outliers} (${(iso.outlier_ratio * 100).toFixed(1)}%)</span>
                </div>`;
            }

            if (results.outlier_results.iqr_method) {
                const totalOutliers = Object.values(results.outlier_results.iqr_method)
                    .reduce((sum, col) => sum + col.n_outliers, 0);
                html += `<div class="metric-row">
                    <span class="metric-label">IQR Method Total Outliers:</span>
                    <span class="metric-value">${totalOutliers}</span>
                </div>`;
            }
        }

        html += '</div>';
        container.innerHTML = html;

        // Display visualizations
        if (results.visualizations) {
            const vizContainer = document.getElementById('statisticalVisualization');
            vizContainer.innerHTML = '';

            for (const [plotName, plotJson] of Object.entries(results.visualizations)) {
                if (plotJson) {
                    const plotDiv = document.createElement('div');
                    plotDiv.style.marginBottom = '20px';
                    vizContainer.appendChild(plotDiv);

                    const plotData = JSON.parse(plotJson);
                    Plotly.newPlot(plotDiv, plotData.data, plotData.layout, {responsive: true});
                }
            }
        }
    }
};

// Event Handlers
const EventHandlers = {
    async loadData() {
        Utils.showLoading(true);

        try {
            const schema = await API.getSchema();
            AppState.schema = schema;
            AppState.dataLoaded = true;

            UIController.updateDataStatus(true);
            UIController.displaySchema(schema);

            Utils.showAlert('Data loaded successfully!', 'success');
        } catch (error) {
            Utils.showAlert(`Error loading data: ${error.message}`, 'danger');
        } finally {
            Utils.showLoading(false);
        }
    },

    async runClustering() {
        if (!AppState.dataLoaded) {
            Utils.showAlert('Please load data first', 'warning');
            return;
        }

        if (AppState.selectedNumericFeatures.length === 0 &&
            AppState.selectedCategoricalFeatures.length === 0) {
            Utils.showAlert('Please select at least one feature', 'warning');
            return;
        }

        Utils.showLoading(true);

        try {
            const algorithm = document.getElementById('clusterAlgorithm').value;
            const nClusters = parseInt(document.getElementById('nClusters').value);
            const scaleNumeric = document.getElementById('scaleNumeric').checked;

            const params = {
                algorithm: algorithm,
                features_numeric: AppState.selectedNumericFeatures,
                features_categorical: AppState.selectedCategoricalFeatures,
                params: { n_clusters: nClusters },
                scale_numeric: scaleNumeric,
                random_state: 42
            };

            const results = await API.runClustering(params);
            AppState.clusteringResults = results;

            UIController.displayClusteringResults(results);
            Utils.showAlert('Clustering completed successfully!', 'success');

        } catch (error) {
            Utils.showAlert(`Error running clustering: ${error.message}`, 'danger');
        } finally {
            Utils.showLoading(false);
        }
    },

    async findOptimalK() {
        if (!AppState.dataLoaded) {
            Utils.showAlert('Please load data first', 'warning');
            return;
        }

        if (AppState.selectedNumericFeatures.length === 0 &&
            AppState.selectedCategoricalFeatures.length === 0) {
            Utils.showAlert('Please select at least one feature', 'warning');
            return;
        }

        Utils.showLoading(true);

        try {
            const algorithm = document.getElementById('clusterAlgorithm').value;
            const scaleNumeric = document.getElementById('scaleNumeric').checked;

            const params = {
                algorithm: algorithm,
                features_numeric: AppState.selectedNumericFeatures,
                features_categorical: AppState.selectedCategoricalFeatures,
                k_min: 2,
                k_max: 10,
                scale_numeric: scaleNumeric,
                random_state: 42
            };

            const results = await API.sweepClustering(params);

            // Display elbow plot
            if (results.visualization) {
                const vizContainer = document.getElementById('clusteringVisualization');
                const plotData = JSON.parse(results.visualization);
                Plotly.newPlot(vizContainer, plotData.data, plotData.layout, {responsive: true});
            }

            // Update optimal k
            if (results.best_k) {
                document.getElementById('nClusters').value = results.best_k;
                Utils.showAlert(`Optimal k found: ${results.best_k}`, 'success');
            }

        } catch (error) {
            Utils.showAlert(`Error finding optimal k: ${error.message}`, 'danger');
        } finally {
            Utils.showLoading(false);
        }
    },

    async runStabilityAnalysis() {
        if (!AppState.dataLoaded) {
            Utils.showAlert('Please load data first', 'warning');
            return;
        }

        Utils.showLoading(true);

        try {
            const algorithm = document.getElementById('clusterAlgorithm').value;
            const nClusters = parseInt(document.getElementById('nClusters').value);
            const scaleNumeric = document.getElementById('scaleNumeric').checked;

            const params = {
                algorithm: algorithm,
                features_numeric: AppState.selectedNumericFeatures,
                features_categorical: AppState.selectedCategoricalFeatures,
                params: { n_clusters: nClusters },
                n_runs: 10,
                subsample_ratio: 0.8,
                scale_numeric: scaleNumeric,
                random_state: 42
            };

            const results = await API.runStabilityAnalysis(params);

            const stabilityHtml = `
                <div class="alert alert-info">
                    <h6>Stability Analysis Results</h6>
                    <p>Mean ARI Score: ${Utils.formatNumber(results.mean_ari)}</p>
                    <p>Std ARI Score: ${Utils.formatNumber(results.std_ari)}</p>
                    <p>Number of comparisons: ${results.n_pairs}</p>
                    <p><strong>Interpretation:</strong> ${
                        results.mean_ari > 0.75 ? 'Highly stable clustering' :
                        results.mean_ari > 0.5 ? 'Moderately stable clustering' :
                        'Unstable clustering - consider different parameters'
                    }</p>
                </div>
            `;

            document.getElementById('clusteringResults').insertAdjacentHTML('afterbegin', stabilityHtml);

        } catch (error) {
            Utils.showAlert(`Error running stability analysis: ${error.message}`, 'danger');
        } finally {
            Utils.showLoading(false);
        }
    },

    async runStatistics() {
        if (!AppState.dataLoaded) {
            Utils.showAlert('Please load data first', 'warning');
            return;
        }

        Utils.showLoading(true);

        try {
            const targetVariable = document.getElementById('targetVariable').value;

            const params = {
                target_column: targetVariable || null
            };

            const results = await API.runStatistics(params);
            AppState.statisticalResults = results;

            UIController.displayStatisticalResults(results);
            Utils.showAlert('Statistical analysis completed!', 'success');

        } catch (error) {
            Utils.showAlert(`Error running statistics: ${error.message}`, 'danger');
        } finally {
            Utils.showLoading(false);
        }
    },

    async exportCSV() {
        if (!AppState.clusteringResults) {
            Utils.showAlert('No clustering results to export', 'warning');
            return;
        }

        Utils.showLoading(true);

        try {
            const results = await API.exportResults({});

            Utils.downloadFile(
                results.csv_data,
                `clustering_results_${Date.now()}.csv`,
                'text/csv'
            );

            Utils.showAlert('CSV exported successfully!', 'success');

        } catch (error) {
            Utils.showAlert(`Error exporting CSV: ${error.message}`, 'danger');
        } finally {
            Utils.showLoading(false);
        }
    },

    async exportJSON() {
        if (!AppState.clusteringResults) {
            Utils.showAlert('No results to export', 'warning');
            return;
        }

        const exportData = {
            timestamp: new Date().toISOString(),
            clustering_results: AppState.clusteringResults,
            statistical_results: AppState.statisticalResults,
            schema: AppState.schema
        };

        Utils.downloadFile(
            JSON.stringify(exportData, null, 2),
            `analysis_results_${Date.now()}.json`,
            'application/json'
        );

        Utils.showAlert('JSON exported successfully!', 'success');
    }
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    // Initialize UI
    UIController.initNavigation();

    // Bind event handlers
    document.getElementById('loadDataBtn').addEventListener('click', EventHandlers.loadData);
    document.getElementById('runClusteringBtn').addEventListener('click', EventHandlers.runClustering);
    document.getElementById('findOptimalKBtn').addEventListener('click', EventHandlers.findOptimalK);
    document.getElementById('stabilityAnalysisBtn').addEventListener('click', EventHandlers.runStabilityAnalysis);
    document.getElementById('runStatisticsBtn').addEventListener('click', EventHandlers.runStatistics);
    document.getElementById('outlierDetectionBtn').addEventListener('click', EventHandlers.runStatistics);
    document.getElementById('exportCSVBtn').addEventListener('click', EventHandlers.exportCSV);
    document.getElementById('exportJSONBtn').addEventListener('click', EventHandlers.exportJSON);

    // Auto-load data on startup
    EventHandlers.loadData();
});