// Work Arrangement Manager Module
// Handles work arrangement impact analysis

const WorkArrangementManager = {
    // Load and display work arrangement analysis
    loadAnalysis: async () => {
        Utils.updateNavStatus('work-arrangement', 'loading');
        try {
            const response = await fetch('/api/work_arrangement_analysis');
            const data = await Utils.parseApiResponse(response);

            if (data.status === 'success') {
                AppState.analysisResults.workArrangement = data.analysis_results;
                WorkArrangementManager.renderResults(data.analysis_results);
                Utils.updateNavStatus('work-arrangement', 'loaded');
                return true;
            }
            return false;
        } catch (error) {
            console.error('Error loading work arrangement analysis:', error);
            Utils.updateNavStatus('work-arrangement', '');
            Utils.handleApiError(error);
            return false;
        }
    },

    // Render all analysis results
    renderResults: (data) => {
        WorkArrangementManager.renderMentalHealthDistribution(data);
        WorkArrangementManager.renderStressBurnoutAnalysis(data);
        WorkArrangementManager.renderDemographicAnalysis(data);
        WorkArrangementManager.renderProductivityMetrics(data);
        WorkArrangementManager.renderKeyInsights(data);
    },

    // Render mental health distribution by work arrangement
    renderMentalHealthDistribution: (data) => {
        if (!data.mental_health_distribution) return;

        const container = document.getElementById('workArrangementContent');
        if (!container) return;

        const distribution = data.mental_health_distribution;
        const arrangements = Object.keys(distribution);
        const mentalHealthStatuses = arrangements.length > 0 ?
            Object.keys(distribution[arrangements[0]] || {}) : [];

        if (mentalHealthStatuses.length === 0) {
            container.innerHTML = '<div class="alert alert-warning">No mental health distribution data available</div>';
            return;
        }

        const traces = mentalHealthStatuses.map((status, index) => {
            const colors = ['#2563eb', '#06b6d4', '#10b981', '#f59e0b', '#ef4444'];
            return {
                x: arrangements,
                y: arrangements.map(arr => distribution[arr][status] || 0),
                name: status,
                type: 'bar',
                marker: { color: colors[index % colors.length] }
            };
        });

        const layout = {
            title: {
                text: 'Mental Health Distribution by Work Arrangement',
                font: { color: '#f8fafc' }
            },
            xaxis: {
                title: 'Work Arrangement',
                titlefont: { color: '#f8fafc' },
                tickfont: { color: '#f8fafc' }
            },
            yaxis: {
                title: 'Percentage',
                titlefont: { color: '#f8fafc' },
                tickfont: { color: '#f8fafc' }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#f8fafc' },
            barmode: 'stack',
            hovermode: 'x unified'
        };

        Plotly.newPlot('workArrangementContent', traces, layout, {responsive: true});
    },

    // Render stress and burnout analysis
    renderStressBurnoutAnalysis: (data) => {
        if (!data.stress_burnout_analysis) return;

        const container = document.getElementById('stressBurnoutContent');
        if (!container) return;

        const analysis = data.stress_burnout_analysis;
        const arrangements = Object.keys(analysis);

        if (arrangements.length === 0) {
            container.innerHTML = '<div class="alert alert-warning">No stress/burnout data available</div>';
            return;
        }

        const wlbScores = arrangements.map(arr => analysis[arr].avg_work_life_balance_score || 0);
        const isolationScores = arrangements.map(arr => analysis[arr].avg_social_isolation_score || 0);

        const traces = [
            {
                x: arrangements,
                y: wlbScores,
                name: 'Work-Life Balance',
                type: 'bar',
                marker: { color: '#10b981' }
            },
            {
                x: arrangements,
                y: isolationScores,
                name: 'Social Isolation',
                type: 'bar',
                marker: { color: '#ef4444' },
                yaxis: 'y2'
            }
        ];

        const layout = {
            title: {
                text: 'Work-Life Balance vs Social Isolation by Arrangement',
                font: { color: '#f8fafc' }
            },
            xaxis: {
                title: 'Work Arrangement',
                titlefont: { color: '#f8fafc' },
                tickfont: { color: '#f8fafc' }
            },
            yaxis: {
                title: 'Work-Life Balance Score',
                titlefont: { color: '#f8fafc' },
                tickfont: { color: '#f8fafc' },
                side: 'left'
            },
            yaxis2: {
                title: 'Social Isolation Score',
                titlefont: { color: '#f8fafc' },
                tickfont: { color: '#f8fafc' },
                overlaying: 'y',
                side: 'right'
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#f8fafc' },
            hovermode: 'x unified'
        };

        Plotly.newPlot('stressBurnoutContent', traces, layout, {responsive: true});
    },

    // Render demographic analysis
    renderDemographicAnalysis: (data) => {
        if (!data.demographic_analysis) return;

        const container = document.getElementById('demographicContent');
        if (!container) return;

        const analysis = data.demographic_analysis;

        let html = '<div class="row">';

        Object.entries(analysis).forEach(([arrangement, demo]) => {
            html += `
                <div class="col-md-6 mb-3">
                    <div class="stats-card">
                        <h6 style="color: var(--text-primary);">${arrangement}</h6>
                        <div class="mb-2">
                            <strong>Average Age:</strong> ${demo.avg_age?.toFixed(1) || 'N/A'}
                        </div>
                        
                        ${demo.gender_distribution ? `
                            <div class="mb-2">
                                <strong>Gender Distribution:</strong><br>
                                ${Object.entries(demo.gender_distribution).map(([gender, pct]) => 
                                    `<span class="badge bg-secondary me-1">${gender}: ${pct.toFixed(1)}%</span>`
                                ).join('')}
                            </div>
                        ` : ''}
                        
                        ${demo.age_groups ? `
                            <div class="mb-2">
                                <strong>Age Groups:</strong><br>
                                ${Object.entries(demo.age_groups).map(([group, count]) => 
                                    `<div class="small">${group}: ${count} people</div>`
                                ).join('')}
                            </div>
                        ` : ''}
                        
                        ${demo.industry_top ? `
                            <div class="mb-2">
                                <strong>Top Industries:</strong><br>
                                ${demo.industry_top.slice(0, 3).map(industry => 
                                    `<span class="badge bg-info me-1">${industry}</span>`
                                ).join('')}
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
        });

        html += '</div>';
        container.innerHTML = html;
    },

    // Render productivity metrics
    renderProductivityMetrics: (data) => {
        if (!data.productivity_metrics) return;

        const container = document.getElementById('productivityContent');
        if (!container) return;

        const metrics = data.productivity_metrics;
        const arrangements = Object.keys(metrics);

        if (arrangements.length === 0) {
            container.innerHTML = '<div class="alert alert-info">No productivity metrics available</div>';
            return;
        }

        // Create radar chart for multi-dimensional comparison
        const traces = arrangements.map((arr, idx) => ({
            type: 'scatterpolar',
            r: [
                metrics[arr].efficiency || 0,
                metrics[arr].collaboration || 0,
                metrics[arr].innovation || 0,
                metrics[arr].satisfaction || 0,
                metrics[arr].retention || 0
            ],
            theta: ['Efficiency', 'Collaboration', 'Innovation', 'Satisfaction', 'Retention'],
            fill: 'toself',
            name: arr,
            marker: { color: ['#2563eb', '#06b6d4', '#10b981', '#f59e0b'][idx % 4] }
        }));

        const layout = {
            title: {
                text: 'Productivity Metrics Comparison',
                font: { color: '#f8fafc' }
            },
            polar: {
                radialaxis: {
                    visible: true,
                    range: [0, 100],
                    tickfont: { color: '#f8fafc' }
                },
                angularaxis: {
                    tickfont: { color: '#f8fafc' }
                },
                bgcolor: 'rgba(0,0,0,0)'
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#f8fafc' },
            showlegend: true
        };

        Plotly.newPlot('productivityContent', traces, layout, {responsive: true});
    },

    // Render key insights and recommendations
    renderKeyInsights: (data) => {
        const container = document.getElementById('keyInsightsContent');
        if (!container) return;

        const insights = data.key_insights || [];
        const recommendations = data.recommendations || [];

        let html = `
            <div class="row">
                <div class="col-md-6">
                    <div class="stats-card">
                        <h5 style="color: var(--text-primary);">
                            <i class="fas fa-lightbulb"></i> Key Insights
                        </h5>
                        ${insights.length > 0 ? `
                            <ul class="mt-3">
                                ${insights.map(insight => `
                                    <li class="mb-2" style="color: var(--text-secondary);">
                                        ${insight}
                                    </li>
                                `).join('')}
                            </ul>
                        ` : '<p class="text-muted">No insights available</p>'}
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="stats-card">
                        <h5 style="color: var(--text-primary);">
                            <i class="fas fa-tasks"></i> Recommendations
                        </h5>
                        ${recommendations.length > 0 ? `
                            <ul class="mt-3">
                                ${recommendations.map(rec => `
                                    <li class="mb-2" style="color: var(--text-secondary);">
                                        ${rec}
                                    </li>
                                `).join('')}
                            </ul>
                        ` : '<p class="text-muted">No recommendations available</p>'}
                    </div>
                </div>
            </div>
        `;

        container.innerHTML = html;
    },

    // Export work arrangement analysis
    exportAnalysis: () => {
        if (!AppState.analysisResults.workArrangement) {
            Utils.showAlert('No analysis results to export', 'warning');
            return;
        }

        const data = {
            timestamp: new Date().toISOString(),
            analysis: AppState.analysisResults.workArrangement
        };

        Utils.downloadJSON(data, `work_arrangement_analysis_${Date.now()}.json`);
        Utils.showAlert('Analysis exported successfully!', 'success');
    }
};