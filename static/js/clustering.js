// Clustering Manager Module
// Handles clustering analysis and visualization

const ClusteringManager = {
    // Setup clustering controls
    setupControls: () => {
        if (!AppState.schemaInfo) return;

        // Setup features grid
        const featuresContainer = document.getElementById('featuresGrid');
        const numericColumns = AppState.schemaInfo.numeric_columns || [];

        const defaultFeatures = ['Age', 'Hours_Per_Week', 'Work_Life_Balance_Score', 'Social_Isolation_Score'];

        // Create grid layout for features
        let featuresHtml = '';
        numericColumns.forEach((col, index) => {
            const isDefault = defaultFeatures.includes(col);
            const colClass = index % 2 === 0 ? 'col-6' : 'col-6';
            featuresHtml += `
                <div class="${colClass}">
                    <div class="form-check form-check-sm">
                        <input class="form-check-input" type="checkbox" value="${col}" id="feature_${col}" ${isDefault ? 'checked' : ''}>
                        <label class="form-check-label" for="feature_${col}" style="color: var(--text-secondary); font-size: 0.85rem;">
                            ${col.replace(/_/g, ' ')}
                        </label>
                    </div>
                </div>
            `;
        });
        featuresContainer.innerHTML = featuresHtml;

        // Setup slider
        const slider = document.getElementById('clustersSlider');
        const valueDisplay = document.getElementById('clustersValue');

        if (slider && valueDisplay) {
            slider.addEventListener('input', (e) => {
                valueDisplay.textContent = e.target.value;
            });
        }

        // Algorithm selection affects k-slider visibility
        const algorithmSelect = document.getElementById('algorithmSelect');
        if (algorithmSelect) {
            algorithmSelect.addEventListener('change', (e) => {
                const sliderContainer = slider?.parentElement;
                if (sliderContainer) {
                    if (e.target.value === 'dbscan') {
                        sliderContainer.style.display = 'none';
                    } else {
                        sliderContainer.style.display = 'block';
                    }
                }
            });
        }

        // Setup buttons
        const runBtn = document.getElementById('runClusteringBtn');
        const optimizeBtn = document.getElementById('optimizeKBtn');

        if (runBtn) {
            runBtn.addEventListener('click', () => {
                ClusteringManager.runClustering();
            });
        }

        if (optimizeBtn) {
            optimizeBtn.addEventListener('click', () => {
                ClusteringManager.findOptimalK();
            });
        }
    },

    // Run clustering analysis
    runClustering: async () => {
        const algorithm = document.getElementById('algorithmSelect')?.value || 'kmeans';
        const k = algorithm === 'dbscan' ? null : document.getElementById('clustersSlider')?.value;
        const selectedFeatures = Array.from(
            document.querySelectorAll('#featuresGrid input:checked')
        ).map(cb => cb.value);

        if (selectedFeatures.length < 2) {
            Utils.showAlert('Please select at least 2 features for clustering', 'warning');
            return;
        }

        Utils.updateNavStatus('clustering', 'loading');

        try {
            const config = {
                k: k,
                features: selectedFeatures,
                algorithm: algorithm
            };

            const result = await DataAnalysis.performClustering(config);

            if (result) {
                AppState.analysisResults.clustering = result;
                ClusteringManager.renderResults(result);
                Utils.updateNavStatus('clustering', 'loaded');
                Utils.showAlert(`${algorithm} clustering completed successfully!`, 'success');
            } else {
                Utils.showAlert('Clustering analysis failed', 'error');
            }
        } catch (error) {
            Utils.handleApiError(error);
            Utils.updateNavStatus('clustering', '');
        }
    },

    // Find optimal K value
    findOptimalK: async () => {
        const selectedFeatures = Array.from(
            document.querySelectorAll('#featuresGrid input:checked')
        ).map(cb => cb.value);

        if (selectedFeatures.length < 2) {
            Utils.showAlert('Please select at least 2 features for clustering', 'warning');
            return;
        }

        // Set algorithm to k-means for optimization
        const algorithmSelect = document.getElementById('algorithmSelect');
        if (algorithmSelect) {
            algorithmSelect.value = 'kmeans';
        }

        await ClusteringManager.runClustering();
    },

    // Render clustering results
    renderResults: (data) => {
        ClusteringManager.renderMainVisualization(data);
        ClusteringManager.renderEvaluationMetrics(data);
        ClusteringManager.renderFeatureContributions(data);
        ClusteringManager.renderFeatureDistributions(data);
        ClusteringManager.renderClusterInsights(data);
    },

    // Render main PCA visualization
    renderMainVisualization: (data) => {
        if (!data.pca_data) return;

        const pcaData = data.pca_data;
        const trace = {
            x: pcaData.x,
            y: pcaData.y,
            mode: 'markers',
            type: 'scatter',
            marker: {
                color: pcaData.clusters,
                colorscale: 'Viridis',
                size: 8,
                colorbar: {
                    title: 'Cluster',
                    titlefont: { color: '#f8fafc' },
                    tickfont: { color: '#f8fafc' }
                }
            },
            text: pcaData.clusters.map(c => `Cluster ${c}`),
            hovertemplate: 'PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>%{text}<extra></extra>'
        };

        const layout = {
            title: {
                text: `Clustering Results (k=${data.optimal_k})`,
                font: { color: '#f8fafc' }
            },
            xaxis: {
                title: `PC1 (${(pcaData.explained_variance[0] * 100).toFixed(1)}%)`,
                titlefont: { color: '#f8fafc' },
                tickfont: { color: '#f8fafc' },
                gridcolor: 'rgba(248, 250, 252, 0.2)'
            },
            yaxis: {
                title: `PC2 (${(pcaData.explained_variance[1] * 100).toFixed(1)}%)`,
                titlefont: { color: '#f8fafc' },
                tickfont: { color: '#f8fafc' },
                gridcolor: 'rgba(248, 250, 252, 0.2)'
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#f8fafc' }
        };

        Plotly.newPlot('clusteringContent', [trace], layout, {responsive: true});
    },

    // Render evaluation metrics
    renderEvaluationMetrics: (data) => {
        const container = document.getElementById('metricsContent');
        if (!container) return;

        let html = `
            <div class="mb-3">
                <h6 style="color: var(--text-primary);">Algorithm: ${data.algorithm || 'k-means'}</h6>
                <div class="mb-2"><strong>Optimal k:</strong> ${data.optimal_k}</div>
            </div>
        `;

        if (data.silhouette_scores && data.silhouette_scores.length > 0) {
            // Create silhouette score chart
            const kValues = data.silhouette_scores.map(item => item[0]);
            const scores = data.silhouette_scores.map(item => item[1]);

            const trace = {
                x: kValues,
                y: scores,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Silhouette Score',
                line: { color: '#06b6d4' },
                marker: { color: '#06b6d4', size: 8 }
            };

            const layout = {
                title: {
                    text: 'Silhouette Score by k',
                    font: { color: '#f8fafc', size: 14 }
                },
                xaxis: {
                    title: 'Number of Clusters (k)',
                    titlefont: { color: '#f8fafc', size: 12 },
                    tickfont: { color: '#f8fafc' }
                },
                yaxis: {
                    title: 'Silhouette Score',
                    titlefont: { color: '#f8fafc', size: 12 },
                    tickfont: { color: '#f8fafc' }
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#f8fafc' },
                height: 250,
                margin: { l: 50, r: 20, t: 40, b: 40 }
            };

            html += '<div id="silhouetteChart" style="height: 250px;"></div>';
            container.innerHTML = html;

            Plotly.newPlot('silhouetteChart', [trace], layout, {responsive: true});
        } else {
            container.innerHTML = html;
        }
    },

    // Render feature contributions
    renderFeatureContributions: (data) => {
        const container = document.getElementById('contributionsContent');
        if (!container || !data.feature_names) return;

        // Mock feature contributions if not provided
        const contributions = {};
        data.feature_names.forEach((feature, idx) => {
            contributions[feature] = Math.random(); // Replace with actual values if available
        });

        const sortedContributions = Object.entries(contributions)
            .sort(([,a], [,b]) => b - a);

        const trace = {
            x: sortedContributions.map(([,value]) => value),
            y: sortedContributions.map(([name,]) => name.replace(/_/g, ' ')),
            type: 'bar',
            orientation: 'h',
            marker: {
                color: 'rgba(37, 99, 235, 0.8)',
                line: {
                    color: 'rgba(37, 99, 235, 1)',
                    width: 1
                }
            }
        };

        const layout = {
            title: {
                text: 'Feature Importance in Clustering',
                font: { color: '#f8fafc' }
            },
            xaxis: {
                title: 'Contribution',
                titlefont: { color: '#f8fafc' },
                tickfont: { color: '#f8fafc' }
            },
            yaxis: {
                titlefont: { color: '#f8fafc' },
                tickfont: { color: '#f8fafc' }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#f8fafc' },
            margin: { l: 150, r: 50, t: 50, b: 50 }
        };

        Plotly.newPlot('contributionsContent', [trace], layout, {responsive: true});
    },

    // Render feature distributions
    renderFeatureDistributions: (data) => {
        const container = document.getElementById('distributionsContent');
        if (!container || !data.feature_names || data.feature_names.length === 0) return;

        // Use first feature for distribution visualization
        const firstFeature = data.feature_names[0];
        const colors = ['#2563eb', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#f97316', '#84cc16'];

        // Create mock distributions for demo (replace with actual data)
        const traces = [];
        for (let i = 0; i < data.optimal_k; i++) {
            traces.push({
                x: Array.from({length: 30}, () => Math.random() * 100),
                type: 'histogram',
                name: `Cluster ${i}`,
                opacity: 0.7,
                marker: { color: colors[i % colors.length] },
                nbinsx: 20
            });
        }

        const layout = {
            title: {
                text: `${firstFeature.replace(/_/g, ' ')} Distribution by Cluster`,
                font: { color: '#f8fafc' }
            },
            xaxis: {
                title: firstFeature.replace(/_/g, ' '),
                titlefont: { color: '#f8fafc' },
                tickfont: { color: '#f8fafc' }
            },
            yaxis: {
                title: 'Count',
                titlefont: { color: '#f8fafc' },
                tickfont: { color: '#f8fafc' }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#f8fafc' },
            barmode: 'overlay'
        };

        Plotly.newPlot('distributionsContent', traces, layout, {responsive: true});
    },

    // Render cluster insights
    renderClusterInsights: (data) => {
        const container = document.getElementById('insightsContent');
        if (!container || !data.cluster_summary) return;

        let html = '<div class="row">';

        Object.entries(data.cluster_summary).forEach(([clusterName, info]) => {
            const clusterNum = clusterName.split('_')[1];
            const percentage = ((info.size / data.clusters.length) * 100).toFixed(1);

            html += `
                <div class="col-md-6 mb-4">
                    <div class="stats-card">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h5 style="color: var(--text-primary);">Cluster ${clusterNum}</h5>
                            <span class="badge bg-primary">${info.size} members (${percentage}%)</span>
                        </div>
                        
                        <div class="mb-3">
                            <h6 style="color: var(--text-primary);">Average Metrics:</h6>
                            <div class="mb-1"><strong>Age:</strong> ${info.avg_age?.toFixed(1) || 'N/A'}</div>
                            <div class="mb-1"><strong>Hours/Week:</strong> ${info.avg_hours?.toFixed(1) || 'N/A'}</div>
                            <div class="mb-1"><strong>Work-Life Balance:</strong> ${info.avg_work_life_balance?.toFixed(2) || 'N/A'}</div>
                            <div class="mb-1"><strong>Social Isolation:</strong> ${info.avg_isolation?.toFixed(2) || 'N/A'}</div>
                        </div>

                        ${info.mental_health_dist && Object.keys(info.mental_health_dist).length > 0 ? `
                            <div class="mb-3">
                                <h6 style="color: var(--text-primary);">Mental Health Distribution:</h6>
                                ${Object.entries(info.mental_health_dist).map(([status, count]) => 
                                    `<div class="mb-1"><strong>${status}:</strong> ${count} people</div>`
                                ).join('')}
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
        });

        html += '</div>';
        container.innerHTML = html;
    }
};