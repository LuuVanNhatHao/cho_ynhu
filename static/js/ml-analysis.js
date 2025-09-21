// ML Analysis Manager Module
// Handles machine learning analysis rendering and display

const MLAnalysisManager = {
    // Render all ML results
    renderResults: (data) => {
        // Check if accuracy is too low
        const bestModelAccuracy = data.models_performance[data.best_model]?.accuracy || 0;

        if (bestModelAccuracy < 0.7) {
            MLAnalysisManager.showLowAccuracyWarning();
            MLAnalysisManager.renderDescriptiveAnalysis(data);
        } else {
            MLAnalysisManager.hideLowAccuracyWarning();
        }

        MLAnalysisManager.renderModelPerformance(data.models_performance, data.best_model);
        MLAnalysisManager.renderFeatureImportance(data.feature_importance, data.best_model);
        MLAnalysisManager.renderModelEvaluation(data.confusion_matrices, data.best_model, data.target_classes);
    },

    // Show low accuracy warning
    showLowAccuracyWarning: () => {
        const alert = document.getElementById('mlPerformanceAlert');
        const altAnalysis = document.getElementById('alternativeAnalysis');
        if (alert) alert.style.display = 'block';
        if (altAnalysis) altAnalysis.style.display = 'block';
    },

    // Hide low accuracy warning
    hideLowAccuracyWarning: () => {
        const alert = document.getElementById('mlPerformanceAlert');
        const altAnalysis = document.getElementById('alternativeAnalysis');
        if (alert) alert.style.display = 'none';
        if (altAnalysis) altAnalysis.style.display = 'none';
    },

    // Render descriptive analysis for low accuracy scenarios
    renderDescriptiveAnalysis: (data) => {
        const container = document.getElementById('descriptiveContent');
        if (!container) return;

        let html = `
            <div class="row">
                <div class="col-md-12">
                    <div class="alert alert-info">
                        <h6><i class="fas fa-info-circle"></i> Alternative Analysis Approach</h6>
                        <p>Since ML model accuracy is below 70%, we recommend focusing on descriptive statistics and correlation analysis instead of predictive modeling.</p>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <h6 style="color: var(--text-primary);">Recommended Actions:</h6>
                    <ul style="color: var(--text-secondary);">
                        <li>Explore correlation patterns in the Visualizations section</li>
                        <li>Use clustering analysis to identify natural groupings</li>
                        <li>Focus on descriptive statistics and trend analysis</li>
                        <li>Consider collecting additional features for better prediction</li>
                    </ul>
                </div>
                <div class="col-md-6">
                    <h6 style="color: var(--text-primary);">Possible Reasons for Low Accuracy:</h6>
                    <ul style="color: var(--text-secondary);">
                        <li>Insufficient features for prediction</li>
                        <li>High variability in mental health outcomes</li>
                        <li>Need for non-linear modeling approaches</li>
                        <li>Data quality or preprocessing issues</li>
                    </ul>
                </div>
            </div>
        `;

        container.innerHTML = html;
    },

    // Render model performance comparison
    renderModelPerformance: (performance, bestModel) => {
        const container = document.getElementById('mlModelsContent');
        if (!container) return;

        let html = '<div class="row">';

        Object.entries(performance).forEach(([modelName, metrics]) => {
            const isBest = modelName === bestModel;
            const accuracyPercentage = (metrics.accuracy * 100).toFixed(2);
            const isLowAccuracy = metrics.accuracy < 0.7;

            html += `
                <div class="col-md-4">
                    <div class="stats-card ${isBest ? 'best-model' : ''}" style="position: relative;">
                        ${isBest ? '<div style="position: absolute; top: 10px; right: 15px; font-size: 1.5rem;">👑</div>' : ''}
                        <h5 style="color: var(--text-primary); margin-bottom: 15px;">${modelName}</h5>
                        <div class="mb-2">
                            <strong>Accuracy:</strong> 
                            <span class="${isLowAccuracy ? 'text-warning' : ''}">${accuracyPercentage}%</span>
                            ${isLowAccuracy ? '<i class="fas fa-exclamation-triangle text-warning ms-1"></i>' : ''}
                        </div>
                        <div class="mb-2">
                            <strong>CV Score:</strong> ${(metrics.cv_mean * 100).toFixed(2)}% ± ${(metrics.cv_std * 100).toFixed(2)}%
                        </div>
                        ${isBest ? '<div class="badge bg-success">Best Model</div>' : ''}
                    </div>
                </div>
            `;
        });

        html += '</div>';
        container.innerHTML = html;
    },

    // Render feature importance chart
    renderFeatureImportance: (featureImportance, bestModel) => {
        const container = document.getElementById('featureImportanceContent');
        if (!container || !featureImportance[bestModel]) return;

        const importance = featureImportance[bestModel];
        const sortedFeatures = Object.entries(importance)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 10);

        const trace = {
            x: sortedFeatures.map(([,value]) => value),
            y: sortedFeatures.map(([name,]) => name.replace(/_/g, ' ')),
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
                text: `Feature Importance - ${bestModel}`,
                font: { color: '#f8fafc' }
            },
            xaxis: {
                title: 'Importance',
                titlefont: { color: '#f8fafc' },
                tickfont: { color: '#f8fafc' },
                gridcolor: 'rgba(248, 250, 252, 0.2)'
            },
            yaxis: {
                title: 'Features',
                titlefont: { color: '#f8fafc' },
                tickfont: { color: '#f8fafc' }
            },
            margin: { l: 150, r: 50, t: 50, b: 50 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#f8fafc' }
        };

        Plotly.newPlot('featureImportanceContent', [trace], layout, {responsive: true});
    },

    // Render confusion matrix
    renderModelEvaluation: (confusionMatrices, bestModel, targetClasses) => {
        const container = document.getElementById('modelEvaluationContent');
        if (!container || !confusionMatrices[bestModel]) return;

        const cm = confusionMatrices[bestModel];

        const trace = {
            z: cm,
            x: targetClasses,
            y: targetClasses,
            type: 'heatmap',
            colorscale: 'Blues',
            showscale: true,
            hoverongaps: false,
            hovertemplate: 'Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
        };

        const layout = {
            title: {
                text: `Confusion Matrix - ${bestModel}`,
                font: { color: '#f8fafc' }
            },
            xaxis: {
                title: 'Predicted',
                titlefont: { color: '#f8fafc' },
                tickfont: { color: '#f8fafc' }
            },
            yaxis: {
                title: 'Actual',
                titlefont: { color: '#f8fafc' },
                tickfont: { color: '#f8fafc' }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#f8fafc' }
        };

        Plotly.newPlot('modelEvaluationContent', [trace], layout, {responsive: true});
    }
};