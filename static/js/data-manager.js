// Data Management Module
// Handles data loading, caching, and overview display

const DataManager = {
    // Auto-load data on page load
    autoLoadData: async () => {
        Utils.showAutoLoadingStatus('loading', 'Auto-loading data...');

        try {
            const response = await fetch('/api/initial_load');
            const data = await Utils.parseApiResponse(response);

            if (data.status === 'success') {
                AppState.dataLoaded = true;
                AppState.schemaInfo = data.schema_info;
                AppState.dataFingerprint = data.data_fingerprint;

                DataManager.updateQuickStats(data.basic_stats);
                DataManager.displayDataOverview(data);

                if (data.initial_plots) {
                    DataManager.displayInitialPlots(data.initial_plots);
                }

                Utils.showAutoLoadingStatus('success', '✅ Data auto-loaded successfully!');
                Utils.showAlert('🎉 Data is ready! You can start analysis now.', 'success');
                Utils.updateNavStatus('dashboard', 'loaded');

                // Store load timestamp
                AppState.dataLoadedAt = new Date();

                return true;
            } else {
                Utils.showAutoLoadingStatus('error', '❌ Failed to auto-load data');
                Utils.showAlert(`⚠️ ${data.message}`, 'warning');
                return false;
            }
        } catch (error) {
            Utils.showAutoLoadingStatus('error', '❌ Connection error');
            Utils.handleApiError(error);
            return false;
        }
    },

    // Reload data manually
    reloadData: async () => {
        Utils.showLoading(true);
        try {
            const response = await fetch('/api/load_data');
            const data = await Utils.parseApiResponse(response);

            if (data.status === 'success') {
                AppState.dataLoaded = true;
                AppState.schemaInfo = data.schema_info;
                AppState.dataFingerprint = data.data_fingerprint;

                // Clear cache
                AppState.analysisResults = {};

                Utils.showAlert(`✅ ${data.message}`, 'success');
                DataManager.updateQuickStats(data.basic_stats);
                DataManager.displayDataOverview(data);

                // Update load timestamp
                AppState.dataLoadedAt = new Date();

                // Clear all visualizations
                if (window.VisualizationManager) {
                    VisualizationManager.clearAllPlots();
                }

                return true;
            }
        } catch (error) {
            Utils.handleApiError(error);
            return false;
        } finally {
            Utils.showLoading(false);
        }
    },

    // Update quick statistics cards with animations
    updateQuickStats: (stats) => {
        const totalRecords = document.getElementById('totalRecords');
        const totalColumns = document.getElementById('totalColumns');
        const missingValues = document.getElementById('missingValues');
        const memoryUsage = document.getElementById('memoryUsage');

        if (totalRecords) Utils.animateValue(totalRecords, 0, stats.total_records);
        if (totalColumns) Utils.animateValue(totalColumns, 0, stats.total_columns);
        if (missingValues) Utils.animateValue(missingValues, 0, stats.missing_values);
        if (memoryUsage) memoryUsage.textContent = stats.memory_usage;

        // Store stats in AppState for later use
        AppState.basicStats = stats;
    },

    // Display data overview table
    displayDataOverview: (data) => {
        const container = document.getElementById('dataOverviewContent');
        if (!container) return;

        const overviewHtml = `
            <div class="row">
                <div class="col-md-6">
                    <h5 style="color: var(--text-primary);">Data Structure</h5>
                    <div class="mb-3">
                        <small class="text-muted">
                            Fingerprint: ${AppState.dataFingerprint || 'N/A'} | 
                            Loaded: ${AppState.dataLoadedAt ? new Date(AppState.dataLoadedAt).toLocaleString() : 'N/A'}
                        </small>
                    </div>
                    <div class="table-responsive">
                        <table class="table table-dark table-striped">
                            <thead>
                                <tr>
                                    <th>Column</th>
                                    <th>Data Type</th>
                                    <th>Unique Values</th>
                                    <th>Missing</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${data.columns.map(col => {
                                    const info = data.schema_info.column_info[col];
                                    const missingPercent = ((info.null_count / data.basic_stats.total_records) * 100).toFixed(1);
                                    return `
                                        <tr>
                                            <td title="${col}">${DataManager.truncateText(col, 20)}</td>
                                            <td><span class="badge bg-primary">${info.dtype}</span></td>
                                            <td>${info.unique_count}</td>
                                            <td>
                                                ${info.null_count} 
                                                ${info.null_count > 0 ? `<small class="text-warning">(${missingPercent}%)</small>` : ''}
                                            </td>
                                        </tr>
                                    `;
                                }).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="col-md-6">
                    <h5 style="color: var(--text-primary);">Sample Data</h5>
                    <div class="mb-3">
                        <button class="btn btn-sm btn-glass" onclick="DataManager.exportSampleData()">
                            <i class="fas fa-download"></i> Export Sample
                        </button>
                    </div>
                    <div class="table-responsive">
                        <table class="table table-dark table-striped">
                            <thead>
                                <tr>
                                    ${data.columns.slice(0, 4).map(col => `
                                        <th title="${col}">${DataManager.truncateText(col, 15)}</th>
                                    `).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                ${data.sample_data.slice(0, 5).map((row, idx) => `
                                    <tr>
                                        ${data.columns.slice(0, 4).map(col => `
                                            <td title="${row[col]}">${DataManager.truncateText(row[col], 15)}</td>
                                        `).join('')}
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                    <div class="text-center mt-2">
                        <small class="text-muted">Showing first 5 rows, 4 columns</small>
                    </div>
                </div>
            </div>
        `;

        container.innerHTML = overviewHtml;
        Utils.updateNavStatus('data-overview', 'loaded');

        // Store sample data for export
        AppState.sampleData = data.sample_data;
    },

    // Display initial plots
    displayInitialPlots: (plots) => {
        if (plots.dashboard_overview) {
            const container = document.getElementById('dashboardChart');
            const plotArea = document.getElementById('dashboardPlotArea');

            if (container && plotArea) {
                container.style.display = 'block';

                try {
                    const plotData = JSON.parse(plots.dashboard_overview);

                    // Add custom config for better interactivity
                    const config = {
                        responsive: true,
                        displayModeBar: true,
                        modeBarButtonsToRemove: ['sendDataToCloud'],
                        displaylogo: false
                    };

                    Plotly.newPlot('dashboardPlotArea', plotData.data, plotData.layout, config);
                } catch (error) {
                    console.error('Error rendering initial plot:', error);
                    plotArea.innerHTML = '<div class="alert alert-warning">Unable to render initial visualization</div>';
                }
            }
        }
    },

    // Truncate text for display
    truncateText: (text, maxLength) => {
        if (!text && text !== 0) return 'N/A';
        const str = text.toString();
        return str.length > maxLength ? str.substring(0, maxLength) + '...' : str;
    },

    // Get column statistics
    getColumnStats: (columnName) => {
        if (!AppState.schemaInfo || !AppState.schemaInfo.column_info[columnName]) {
            return null;
        }
        return AppState.schemaInfo.column_info[columnName];
    },

    // Get all numeric columns
    getNumericColumns: () => {
        if (!AppState.schemaInfo) return [];
        return AppState.schemaInfo.numeric_columns || [];
    },

    // Get all categorical columns
    getCategoricalColumns: () => {
        if (!AppState.schemaInfo) return [];
        return AppState.schemaInfo.categorical_columns || [];
    },

    // Check if data is loaded
    isDataLoaded: () => {
        return AppState.dataLoaded;
    },

    // Get data fingerprint
    getDataFingerprint: () => {
        return AppState.dataFingerprint;
    },

    // Get data load timestamp
    getDataLoadTime: () => {
        return AppState.dataLoadedAt;
    },

    // Export sample data as CSV
    exportSampleData: () => {
        if (!AppState.sampleData) {
            Utils.showAlert('No sample data available to export', 'warning');
            return;
        }

        try {
            // Convert sample data to CSV format
            const headers = Object.keys(AppState.sampleData[0]);
            const csvContent = [
                headers.join(','),
                ...AppState.sampleData.map(row =>
                    headers.map(header => {
                        const value = row[header];
                        // Escape values containing commas or quotes
                        if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
                            return `"${value.replace(/"/g, '""')}"`;
                        }
                        return value;
                    }).join(',')
                )
            ].join('\n');

            // Create and download file
            const blob = new Blob([csvContent], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `sample_data_${Date.now()}.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            Utils.showAlert('Sample data exported successfully!', 'success');
        } catch (error) {
            console.error('Error exporting sample data:', error);
            Utils.showAlert('Failed to export sample data', 'error');
        }
    },

    // Refresh data if stale (optional feature)
    checkDataFreshness: () => {
        if (!AppState.dataLoadedAt) return;

        const now = new Date();
        const loadTime = new Date(AppState.dataLoadedAt);
        const hoursSinceLoad = (now - loadTime) / (1000 * 60 * 60);

        if (hoursSinceLoad > 1) {
            const refreshBtn = document.getElementById('reloadDataBtn');
            if (refreshBtn) {
                refreshBtn.classList.add('pulse-animation');
                refreshBtn.title = 'Data may be stale. Consider refreshing.';
            }
        }
    },

    // Get data summary for display
    getDataSummary: () => {
        if (!AppState.basicStats || !AppState.schemaInfo) return null;

        return {
            records: AppState.basicStats.total_records,
            columns: AppState.basicStats.total_columns,
            missing: AppState.basicStats.missing_values,
            memory: AppState.basicStats.memory_usage,
            fingerprint: AppState.dataFingerprint,
            loadedAt: AppState.dataLoadedAt,
            numericCols: AppState.schemaInfo.numeric_columns?.length || 0,
            categoricalCols: AppState.schemaInfo.categorical_columns?.length || 0
        };
    }
};

// Check data freshness periodically
setInterval(() => {
    if (AppState.dataLoaded) {
        DataManager.checkDataFreshness();
    }
}, 60000); // Check every minute