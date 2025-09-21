// Visualization Manager Module
// Handles all chart rendering and filtering

const VisualizationManager = {
    // Setup filter controls
    setupFilters: () => {
        if (!AppState.schemaInfo) return;

        const regionFilter = document.getElementById('regionFilter');
        const industryFilter = document.getElementById('industryFilter');
        const mentalHealthFilter = document.getElementById('mentalHealthFilter');

        // Populate filter options from schema info
        const filterMappings = {
            'Region': regionFilter,
            'Industry': industryFilter,
            'Mental_Health_Status': mentalHealthFilter
        };

        Object.entries(filterMappings).forEach(([columnName, selectElement]) => {
            const info = AppState.schemaInfo.column_info[columnName];
            if (info && info.sample_values && selectElement) {
                info.sample_values.forEach(value => {
                    const option = document.createElement('option');
                    option.value = value;
                    option.textContent = value;
                    selectElement.appendChild(option);
                });
            }
        });

        // Add event listener for apply filters button
        const applyBtn = document.getElementById('applyFiltersBtn');
        if (applyBtn) {
            applyBtn.addEventListener('click', () => {
                VisualizationManager.applyFilters();
            });
        }
    },

    // Apply filters to visualizations
    applyFilters: async () => {
        const filters = {
            region: document.getElementById('regionFilter').value,
            industry: document.getElementById('industryFilter').value,
            mental_health_status: document.getElementById('mentalHealthFilter').value
        };

        const params = new URLSearchParams();
        Object.entries(filters).forEach(([key, value]) => {
            if (value && value !== 'all') {
                params.append(key, value);
            }
        });

        try {
            const response = await fetch('/api/visualizations?' + params.toString());
            const data = await Utils.parseApiResponse(response);

            if (data.status === 'success') {
                VisualizationManager.renderAllPlots(data.plots);
                Utils.showAlert('Filters applied successfully', 'success', 3000);
            }
        } catch (error) {
            Utils.handleApiError(error);
        }
    },

    // Render all plots
    renderAllPlots: (plots) => {
        const plotMappings = {
            'dashboard_overview': 'dashboardOverviewPlot',
            'scatter_3d': 'scatter3dPlot',
            'correlation_heatmap': 'correlationPlot',
            'sunburst': 'sunburstPlot',
            'violin_arrangement': 'violinPlot'
        };

        Object.entries(plots).forEach(([plotKey, plotData]) => {
            const containerId = plotMappings[plotKey];

            if (containerId && plotData) {
                VisualizationManager.renderPlot(containerId, plotData);
            }
        });
    },

    // Render single plot
    renderPlot: (containerId, plotData) => {
        const container = document.getElementById(containerId);
        if (!container) return;

        try {
            const data = JSON.parse(plotData);

            // Clear existing plot
            Plotly.purge(containerId);

            // Create new plot with custom config
            const config = {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['sendDataToCloud'],
                displaylogo: false,
                toImageButtonOptions: {
                    format: 'png',
                    filename: `${containerId}_export`,
                    height: 600,
                    width: 800,
                    scale: 1
                }
            };

            Plotly.newPlot(containerId, data.data, data.layout, config);

            // Store plot instance
            AppState.chartInstances[containerId] = true;
        } catch (error) {
            console.error('Error rendering plot:', containerId, error);
            container.innerHTML = '<div class="alert alert-danger">Error rendering visualization</div>';
        }
    },

    // Update plot layout for dark theme
    updatePlotTheme: (plotId) => {
        const plotElement = document.getElementById(plotId);
        if (!plotElement) return;

        const update = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#f8fafc' },
            xaxis: {
                gridcolor: 'rgba(248, 250, 252, 0.2)',
                tickfont: { color: '#f8fafc' }
            },
            yaxis: {
                gridcolor: 'rgba(248, 250, 252, 0.2)',
                tickfont: { color: '#f8fafc' }
            }
        };

        Plotly.relayout(plotId, update);
    },

    // Export plot as image
    exportPlot: (plotId, format = 'png') => {
        Plotly.downloadImage(plotId, {
            format: format,
            height: 600,
            width: 800,
            filename: `${plotId}_export`
        });
    },

    // Clear all visualizations
    clearAllPlots: () => {
        Object.keys(AppState.chartInstances).forEach(plotId => {
            Plotly.purge(plotId);
        });
        AppState.chartInstances = {};
    },

    // Resize plots when window resizes
    resizePlots: Utils.debounce(() => {
        Object.keys(AppState.chartInstances).forEach(plotId => {
            const element = document.getElementById(plotId);
            if (element) {
                Plotly.Plots.resize(element);
            }
        });
    }, 250)
};

// Add window resize listener for responsive plots
window.addEventListener('resize', VisualizationManager.resizePlots);