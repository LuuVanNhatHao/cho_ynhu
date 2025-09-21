// Clustering & Personas JavaScript - Complete Version
let currentClusters = null;
let currentPersonas = null;
let currentCharts = null;

$(document).ready(function() {
    initializePage();
});

function initializePage() {
    // Setup slider
    $('#n-clusters').on('input', function() {
        $('#n-clusters-display').text($(this).val());
    });

    // Initialize multi-select if Select2 is available
    if (typeof $.fn.select2 !== 'undefined') {
        $('#cluster-features').select2({
            placeholder: 'Select features',
            width: '100%'
        });
    }

    // Add loading spinner styles if not exists
    if ($('#loading-styles').length === 0) {
        const style = `
            <style id="loading-styles">
            .loading-spinner {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(255,255,255,0.95);
                backdrop-filter: blur(10px);
                z-index: 9999;
                display: none;
                align-items: center;
                justify-content: center;
            }
            
            .loading-spinner.active {
                display: flex;
            }
            
            .spinner-overlay {
                background: white;
                padding: 40px;
                border-radius: 20px;
                text-align: center;
                box-shadow: 0 15px 60px rgba(0,0,0,0.15);
            }
            
            .spinner-overlay .spinner-border {
                width: 3rem;
                height: 3rem;
                margin-bottom: 20px;
            }
            </style>
        `;
        $('head').append(style);
    }

    // Add loading spinner if not exists
    if ($('.loading-spinner').length === 0) {
        $('body').append(`
            <div class="loading-spinner">
                <div class="spinner-overlay">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <div class="fw-bold">Analyzing Data...</div>
                    <div class="text-muted small mt-2">This may take a few moments</div>
                </div>
            </div>
        `);
    }
}

function performClustering() {
    const nClusters = parseInt($('#n-clusters').val());
    const method = $('#cluster-method').val();
    const features = $('#cluster-features').val();

    // Validation
    if (!features || features.length === 0) {
        showAlert('Please select at least one feature for clustering', 'warning');
        return;
    }

    if (nClusters < 2 || nClusters > 10) {
        showAlert('Number of clusters must be between 2 and 10', 'warning');
        return;
    }

    showLoading('Performing clustering analysis...');

    const params = {
        n_clusters: nClusters,
        method: method,
        features: features
    };

    fetch('/api/clusters/compute', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify(params)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }

        if (!data.success) {
            throw new Error('Clustering failed - no success flag');
        }

        // Store results globally
        currentClusters = data;
        currentPersonas = data.personas || [];
        currentCharts = data.charts || {};

        // Update UI
        updateClusteringUI(data);

        // Show tabs and hide initial message
        $('#initial-message').hide();
        $('#clusterTabs').show();
        $('#metrics-row').show();
        $('#export-section').show();

        showAlert(`Successfully created ${nClusters} personas using ${method}`, 'success');
    })
    .catch(error => {
        console.error('Clustering error:', error);
        showAlert(`Clustering failed: ${error.message}`, 'danger');
    })
    .finally(() => {
        hideLoading();
    });
}

function updateClusteringUI(data) {
    try {
        // Update metrics cards
        updateMetricsCards(data);

        // Update all visualizations
        updateVisualizations(data);

        // Generate persona cards
        generatePersonaCards(data.personas || []);

        // Update cluster metrics table
        updateMetricsTable(data.personas || []);

        // Populate driver selector
        updateDriverSelector(data.n_clusters);

    } catch (error) {
        console.error('Error updating UI:', error);
        showAlert('Failed to update interface', 'warning');
    }
}

function updateMetricsCards(data) {
    // Silhouette score with animation
    const silhouetteScore = data.silhouette_score || 0;
    animateNumber('#silhouette-score', silhouetteScore.toFixed(3));

    // Number of personas
    animateNumber('#n-personas', data.n_clusters || 0);

    // Largest cluster percentage
    if (data.personas && data.personas.length > 0) {
        const maxSize = Math.max(...data.personas.map(p => p.size || 0));
        const totalSize = data.personas.reduce((sum, p) => sum + (p.size || 0), 0);
        const percentage = totalSize > 0 ? (maxSize / totalSize * 100).toFixed(1) : '0';
        animateNumber('#largest-cluster', percentage + '%');
    } else {
        $('#largest-cluster').text('N/A');
    }
}

function animateNumber(selector, endValue) {
    const element = $(selector);
    const startValue = parseFloat(element.text()) || 0;
    const duration = 1000;
    const steps = 60;
    const increment = (parseFloat(endValue) - startValue) / steps;

    let current = startValue;
    let step = 0;

    const timer = setInterval(() => {
        step++;
        current += increment;

        if (step >= steps) {
            element.text(endValue);
            clearInterval(timer);
        } else {
            const displayValue = typeof endValue === 'string' && endValue.includes('%')
                ? current.toFixed(1) + '%'
                : current.toFixed(3);
            element.text(displayValue);
        }
    }, duration / steps);
}

function updateVisualizations(data) {
    if (!data.charts) {
        console.warn('No chart data available');
        return;
    }

    // UMAP visualization
    if (data.charts.umap_plot) {
        try {
            Plotly.newPlot('umap-chart',
                data.charts.umap_plot.data || [],
                {
                    ...data.charts.umap_plot.layout,
                    height: 600,
                    title: {
                        text: 'Employee Clusters - UMAP Projection',
                        font: { size: 18, family: 'Arial, sans-serif' }
                    },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)'
                },
                {responsive: true, displayModeBar: false}
            );
        } catch (error) {
            console.error('UMAP chart error:', error);
            $('#umap-chart').html('<div class="alert alert-warning">Unable to display UMAP visualization</div>');
        }
    }

    // Cluster profiles radar chart
    if (data.charts.profile) {
        try {
            Plotly.newPlot('radar-chart',
                data.charts.profile.data || [],
                {
                    ...data.charts.profile.layout,
                    height: 600,
                    title: {
                        text: 'Cluster Profile Comparison',
                        font: { size: 18, family: 'Arial, sans-serif' }
                    },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)'
                },
                {responsive: true, displayModeBar: false}
            );
        } catch (error) {
            console.error('Radar chart error:', error);
            $('#radar-chart').html('<div class="alert alert-warning">Unable to display profile chart</div>');
        }
    }

    // Distribution charts
    if (data.charts.distribution) {
        try {
            const distData = data.charts.distribution.data || [];

            // Cluster sizes pie chart
            if (distData.length > 0 && distData[0].type === 'pie') {
                Plotly.newPlot('cluster-sizes-chart',
                    [distData[0]],
                    {
                        title: 'Cluster Size Distribution',
                        height: 400,
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)'
                    },
                    {responsive: true, displayModeBar: false}
                );
            }

            // Characteristics comparison
            if (distData.length > 1) {
                Plotly.newPlot('cluster-characteristics-chart',
                    distData.slice(1),
                    {
                        title: 'Cluster Characteristics',
                        height: 400,
                        barmode: 'group',
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)'
                    },
                    {responsive: true, displayModeBar: false}
                );
            }
        } catch (error) {
            console.error('Distribution chart error:', error);
        }
    }
}

function generatePersonaCards(personas) {
    let html = '';
    const icons = ['fa-user-tie', 'fa-user-cog', 'fa-laptop-house', 'fa-user-graduate', 'fa-user-check', 'fa-user-ninja', 'fa-user-astronaut', 'fa-user-doctor'];
    const colors = ['primary', 'success', 'info', 'warning', 'danger', 'secondary', 'dark', 'light'];

    personas.forEach((persona, index) => {
        const icon = icons[index % icons.length];
        const color = colors[index % colors.length];

        html += `
        <div class="col-md-6 col-lg-4 mb-4">
            <div class="card persona-card h-100" data-persona-id="${persona.id || index}">
                <span class="persona-badge badge bg-${color} position-absolute" style="top: 15px; right: 15px; z-index: 10;">
                    ${persona.percentage || '0'}%
                </span>
                <div class="card-body text-center p-4">
                    <div class="persona-icon text-${color} mb-3">
                        <i class="fas ${icon}" style="font-size: 3rem;"></i>
                    </div>
                    <h4 class="card-title fw-bold mb-2">${persona.name || 'Unknown Persona'}</h4>
                    <p class="text-muted mb-3">${persona.size || 0} employees</p>
                    
                    <div class="mb-4">
                        <span class="metric-badge bg-light text-dark me-1 mb-2 d-inline-block px-3 py-1 rounded-pill">
                            <i class="fas fa-birthday-cake me-1"></i> ${persona.typical_age || 'N/A'}
                        </span>
                        <span class="metric-badge bg-light text-dark me-1 mb-2 d-inline-block px-3 py-1 rounded-pill">
                            <i class="fas fa-clock me-1"></i> ${persona.work_hours || 'N/A'}
                        </span>
                        <span class="metric-badge bg-light text-dark mb-2 d-inline-block px-3 py-1 rounded-pill">
                            <i class="fas fa-balance-scale me-1"></i> ${persona.wlb_status || 'N/A'}
                        </span>
                    </div>
                    
                    <hr class="my-3">
                    
                    <div class="text-start">
                        <div class="mb-2">
                            <strong class="text-muted">Industry:</strong> 
                            <span class="fw-bold">${persona.primary_industry || 'Various'}</span>
                        </div>
                        <div class="mb-2">
                            <strong class="text-muted">Work Style:</strong> 
                            <span class="fw-bold">${persona.work_arrangement || 'Mixed'}</span>
                        </div>
                        <div class="mb-2">
                            <strong class="text-muted">Mental Health:</strong> 
                            <span class="fw-bold">${persona.mental_health || 'Average'}</span>
                        </div>
                        <div class="mb-3">
                            <strong class="text-muted">Burnout Risk:</strong> 
                            <span class="badge bg-${getBurnoutColor(persona.burnout_risk)} ms-2">
                                ${persona.burnout_risk || 'Unknown'}
                            </span>
                        </div>
                    </div>
                    
                    <div class="mt-3 mb-4">
                        <p class="small text-muted fst-italic">${persona.description || 'No description available'}</p>
                    </div>
                    
                    <div class="d-grid gap-2">
                        <button class="btn btn-outline-${color}" onclick="viewPersonaDetails(${persona.id || index})">
                            <i class="fas fa-chart-line me-2"></i>View Details
                        </button>
                    </div>
                </div>
            </div>
        </div>
        `;
    });

    $('#personas-container').html(html);

    // Add animation delay to cards
    $('.persona-card').each(function(index) {
        $(this).css('animation-delay', (index * 0.1) + 's');
        $(this).addClass('animate__animated animate__fadeInUp');
    });
}

function getBurnoutColor(level) {
    if (!level) return 'secondary';

    switch(level.toLowerCase()) {
        case 'low': return 'success';
        case 'medium':
        case 'moderate': return 'warning';
        case 'high': return 'danger';
        case 'very high':
        case 'critical': return 'danger';
        default: return 'secondary';
    }
}

function updateMetricsTable(personas) {
    let html = '';

    personas.forEach((persona, index) => {
        html += `
        <tr class="table-row-hover">
            <td>
                <div class="d-flex align-items-center">
                    <div class="bg-primary rounded-circle me-3" style="width: 12px; height: 12px;"></div>
                    <strong>Cluster ${persona.id || index}</strong>
                </div>
            </td>
            <td><span class="fw-bold">${persona.size || 0}</span></td>
            <td>${persona.typical_age || 'N/A'}</td>
            <td>${persona.work_hours || 'N/A'}</td>
            <td>${persona.wlb_status || 'N/A'}</td>
            <td>${persona.isolation_level || 'N/A'}</td>
            <td>${persona.primary_industry || 'Various'}</td>
            <td>
                <span class="badge bg-${getBurnoutColor(persona.burnout_risk)}">
                    ${persona.burnout_risk || 'Unknown'}
                </span>
            </td>
        </tr>
        `;
    });

    $('#metrics-tbody').html(html);
}

function updateDriverSelector(nClusters) {
    let options = '';
    for (let i = 0; i < nClusters; i++) {
        const persona = currentPersonas && currentPersonas[i] ? currentPersonas[i] : null;
        const name = persona ? persona.name : `Cluster ${i}`;
        options += `<option value="${i}">${name}</option>`;
    }
    $('#driver-cluster-select').html(options);
}

function analyzeDrivers() {
    const clusterId = parseInt($('#driver-cluster-select').val());

    if (isNaN(clusterId)) {
        showAlert('Please select a valid cluster', 'warning');
        return;
    }

    showLoading('Analyzing key drivers...');

    fetch('/api/clusters/drivers', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify({cluster_id: clusterId})
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }

        // Display chart
        if (data.chart) {
            Plotly.newPlot('drivers-chart',
                data.chart.data || [],
                {
                    ...data.chart.layout,
                    height: 500,
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)'
                },
                {responsive: true, displayModeBar: false}
            );
        }

        // Display stats
        displayDriverStats(data.drivers);

        showAlert('Driver analysis completed', 'success');
    })
    .catch(error => {
        console.error('Error analyzing drivers:', error);
        showAlert(`Failed to analyze drivers: ${error.message}`, 'danger');
    })
    .finally(() => {
        hideLoading();
    });
}

function displayDriverStats(drivers) {
    if (!drivers || !drivers.drivers) {
        $('#driver-stats').html('<div class="alert alert-info">No driver data available</div>');
        return;
    }

    let statsHtml = `
        <div class="card">
            <div class="card-header bg-light">
                <h6 class="mb-0"><i class="fas fa-key me-2"></i>Key Characteristics</h6>
            </div>
            <div class="card-body">
                <ul class="list-unstyled mb-3">
    `;

    drivers.drivers.slice(0, 5).forEach((driver, index) => {
        const diff = driver.difference_pct > 0 ? '+' : '';
        const color = driver.difference_pct > 0 ? 'success' : 'danger';

        statsHtml += `
            <li class="mb-2">
                <div class="d-flex justify-content-between align-items-center">
                    <span class="fw-bold">${driver.feature}:</span>
                    <span class="badge bg-${color}">${diff}${driver.difference_pct.toFixed(1)}%</span>
                </div>
            </li>
        `;
    });

    statsHtml += `
                </ul>
                <hr>
                <div class="text-center">
                    <small class="text-muted">
                        <i class="fas fa-chart-line me-1"></i>
                        Model Accuracy: <strong>${((drivers.model_score || 0) * 100).toFixed(1)}%</strong>
                    </small>
                </div>
            </div>
        </div>
    `;

    $('#driver-stats').html(statsHtml);
}

function viewPersonaDetails(personaId) {
    // Switch to drivers tab and analyze
    $('#clusterTabs a[href="#drivers"]').tab('show');
    $('#driver-cluster-select').val(personaId);

    // Animate tab switching
    setTimeout(() => {
        analyzeDrivers();
    }, 300);
}

function exportPersonas() {
    if (!currentPersonas || currentPersonas.length === 0) {
        showAlert('No personas to export. Run clustering first.', 'warning');
        return;
    }

    showLoading('Preparing export...');

    fetch('/api/export/personas')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `personas_${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            showAlert('Personas exported successfully', 'success');
        })
        .catch(error => {
            console.error('Export error:', error);
            showAlert(`Failed to export personas: ${error.message}`, 'danger');
        })
        .finally(() => {
            hideLoading();
        });
}

function saveModel() {
    if (!currentClusters) {
        showAlert('No model to save. Run clustering first.', 'warning');
        return;
    }

    showAlert('Model configuration saved to session', 'success');

    // Store in session storage (if available) or local variable
    try {
        const modelData = {
            clusters: currentClusters,
            personas: currentPersonas,
            timestamp: new Date().toISOString(),
            config: {
                n_clusters: currentClusters.n_clusters,
                method: $('#cluster-method').val(),
                features: $('#cluster-features').val()
            }
        };

        // Store in memory for this session
        window.savedModel = modelData;

        console.log('Model saved:', modelData);
    } catch (error) {
        console.error('Failed to save model:', error);
        showAlert('Failed to save model configuration', 'warning');
    }
}

function compareModels() {
    showAlert('Comparing clustering methods...', 'info');
    showLoading('Running comparison analysis...');

    const methods = ['kmeans', 'gaussian_mixture', 'hierarchical'];
    const nClusters = parseInt($('#n-clusters').val());
    const features = $('#cluster-features').val();

    Promise.all(methods.map(method =>
        fetch('/api/clusters/compute', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                n_clusters: nClusters,
                method: method,
                features: features
            })
        }).then(r => r.json()).catch(e => ({error: e.message, method}))
    ))
    .then(results => {
        let comparisonHtml = `
            <div class="card">
                <div class="card-header bg-info text-white">
                    <h5 class="mb-0"><i class="fas fa-balance-scale me-2"></i>Method Comparison Results</h5>
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-hover">
                            <thead>
                                <tr>
                                    <th>Method</th>
                                    <th>Silhouette Score</th>
                                    <th>Status</th>
                                    <th>Recommendation</th>
                                </tr>
                            </thead>
                            <tbody>
        `;

        let bestScore = -1;
        let bestMethod = '';

        results.forEach((result, index) => {
            const method = methods[index];
            let statusBadge, recommendation, score = 'N/A';

            if (result.error) {
                statusBadge = '<span class="badge bg-danger">Error</span>';
                recommendation = 'Method failed to execute';
            } else {
                score = result.silhouette_score.toFixed(3);
                const scoreFloat = parseFloat(score);

                if (scoreFloat > bestScore) {
                    bestScore = scoreFloat;
                    bestMethod = method;
                }

                if (scoreFloat > 0.7) {
                    statusBadge = '<span class="badge bg-success">Excellent</span>';
                    recommendation = 'Highly recommended';
                } else if (scoreFloat > 0.5) {
                    statusBadge = '<span class="badge bg-warning">Good</span>';
                    recommendation = 'Acceptable clustering';
                } else {
                    statusBadge = '<span class="badge bg-danger">Poor</span>';
                    recommendation = 'Not recommended';
                }
            }

            comparisonHtml += `
                <tr>
                    <td><strong>${method.replace('_', ' ').toUpperCase()}</strong></td>
                    <td>${score}</td>
                    <td>${statusBadge}</td>
                    <td>${recommendation}</td>
                </tr>
            `;
        });

        comparisonHtml += `
                            </tbody>
                        </table>
                    </div>
                    <div class="mt-3 p-3 bg-light rounded">
                        <strong><i class="fas fa-trophy text-warning me-2"></i>Best Method:</strong> 
                        ${bestMethod.replace('_', ' ').toUpperCase()} (Score: ${bestScore.toFixed(3)})
                    </div>
                </div>
            </div>
        `;

        $('#driver-stats').html(comparisonHtml);

        // Switch to drivers tab to show results
        $('#clusterTabs a[href="#drivers"]').tab('show');

        showAlert('Model comparison completed. Check Key Drivers tab for results.', 'success');
    })
    .catch(error => {
        console.error('Comparison error:', error);
        showAlert(`Failed to compare models: ${error.message}`, 'danger');
    })
    .finally(() => {
        hideLoading();
    });
}

function generateReport() {
    if (!currentClusters || !currentPersonas) {
        showAlert('No clustering results to report. Run clustering first.', 'warning');
        return;
    }

    showAlert('Generating comprehensive report...', 'info');

    const reportContent = {
        title: 'Employee Personas & Clustering Analysis Report',
        generated_date: new Date().toLocaleDateString(),
        generated_time: new Date().toLocaleTimeString(),
        analysis_summary: {
            method: $('#cluster-method').val(),
            n_clusters: currentClusters.n_clusters,
            silhouette_score: currentClusters.silhouette_score,
            total_employees: currentPersonas.reduce((sum, p) => sum + (p.size || 0), 0),
            features_used: $('#cluster-features').val()
        },
        personas: currentPersonas.map(persona => ({
            ...persona,
            recommendations: generatePersonaRecommendations(persona)
        })),
        overall_recommendations: generateOverallRecommendations(),
        methodology: {
            clustering_algorithm: $('#cluster-method').val(),
            feature_selection: $('#cluster-features').val(),
            validation_metric: 'Silhouette Score',
            score_interpretation: interpretSilhouetteScore(currentClusters.silhouette_score)
        }
    };

    // Download as JSON
    const blob = new Blob([JSON.stringify(reportContent, null, 2)], {type: 'application/json'});
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `clustering_report_${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);

    showAlert('Comprehensive report generated and downloaded', 'success');
}

function generatePersonaRecommendations(persona) {
    const recommendations = [];

    if (!persona) return recommendations;

    // Burnout recommendations
    if (persona.burnout_risk === 'High' || persona.burnout_risk === 'Very High') {
        recommendations.push({
            category: 'Burnout Prevention',
            actions: [
                'Implement mandatory wellness breaks',
                'Review and redistribute workload',
                'Provide stress management resources',
                'Consider flexible working arrangements'
            ]
        });
    }

    // Isolation recommendations
    if (persona.isolation_level === 'Highly Isolated' || persona.isolation_level === 'Isolated') {
        recommendations.push({
            category: 'Social Connection',
            actions: [
                'Organize regular team building activities',
                'Implement mentorship programs',
                'Create virtual coffee chat sessions',
                'Encourage cross-departmental collaboration'
            ]
        });
    }

    // Work-life balance recommendations
    if (persona.wlb_status === 'Poor Balance' || persona.wlb_status === 'Struggling') {
        recommendations.push({
            category: 'Work-Life Balance',
            actions: [
                'Introduce flexible working hours',
                'Provide work-life balance training',
                'Encourage use of vacation time',
                'Set clear boundaries for after-hours communication'
            ]
        });
    }

    return recommendations;
}

function generateOverallRecommendations() {
    const recommendations = [
        {
            category: 'Strategic Initiatives',
            items: [
                'Develop persona-specific management approaches',
                'Create targeted wellness programs',
                'Implement data-driven HR policies',
                'Regular persona analysis updates'
            ]
        },
        {
            category: 'Management Training',
            items: [
                'Train managers on persona characteristics',
                'Develop persona-based communication strategies',
                'Create individualized development plans',
                'Monitor persona evolution over time'
            ]
        }
    ];

    return recommendations;
}

function interpretSilhouetteScore(score) {
    if (score > 0.7) return 'Excellent clustering - distinct, well-separated groups';
    if (score > 0.5) return 'Good clustering - reasonable group separation';
    if (score > 0.3) return 'Acceptable clustering - some overlap between groups';
    return 'Poor clustering - groups may be artificially forced';
}

// Utility functions
function showAlert(message, type = 'info') {
    const alertClass = type === 'danger' ? 'alert-danger' :
                      type === 'warning' ? 'alert-warning' :
                      type === 'success' ? 'alert-success' :
                      type === 'info' ? 'alert-info' : 'alert-primary';

    const iconClass = type === 'danger' ? 'fa-exclamation-triangle' :
                     type === 'warning' ? 'fa-exclamation-circle' :
                     type === 'success' ? 'fa-check-circle' :
                     type === 'info' ? 'fa-info-circle' : 'fa-bell';

    const alertHtml = `
        <div class="alert ${alertClass} alert-dismissible fade show shadow-sm" role="alert">
            <i class="fas ${iconClass} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
    `;

    // Find or create alert container
    let alertContainer = $('#alert-container');
    if (alertContainer.length === 0) {
        $('body').prepend(`
            <div id="alert-container" 
                 style="position: fixed; top: 80px; right: 20px; z-index: 9999; max-width: 400px;">
            </div>
        `);
        alertContainer = $('#alert-container');
    }

    // Add the alert
    alertContainer.append(alertHtml);

    // Auto-hide after 5 seconds (except for errors which stay longer)
    const hideDelay = type === 'danger' ? 8000 : 5000;
    setTimeout(() => {
        alertContainer.find('.alert').last().alert('close');
    }, hideDelay);

    // Clean up old alerts (keep max 3)
    const alerts = alertContainer.find('.alert');
    if (alerts.length > 3) {
        alerts.first().alert('close');
    }
}

function showLoading(message = 'Processing...') {
    $('.loading-spinner .spinner-overlay div:last-child').text(message);
    $('.loading-spinner').addClass('active').fadeIn(200);
}

function hideLoading() {
    $('.loading-spinner').removeClass('active').fadeOut(200);
}

// Initialize tooltips and other UI enhancements
function initializeUIEnhancements() {
    // Initialize tooltips if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // Add smooth scrolling for anchor links
    $('a[href^="#"]').on('click', function(e) {
        e.preventDefault();
        const target = $($(this).attr('href'));
        if (target.length) {
            $('html, body').animate({
                scrollTop: target.offset().top - 100
            }, 500);
        }
    });

    // Add loading states to buttons
    $('.btn').on('click', function() {
        const btn = $(this);
        if (!btn.hasClass('no-loading')) {
            const originalText = btn.html();
            btn.data('original-text', originalText);
            btn.html('<i class="fas fa-spinner fa-spin me-2"></i>Processing...');
            btn.prop('disabled', true);

            // Re-enable after 3 seconds as fallback
            setTimeout(() => {
                if (btn.data('original-text')) {
                    btn.html(btn.data('original-text'));
                    btn.prop('disabled', false);
                }
            }, 3000);
        }
    });
}

// Restore button states
function restoreButtonStates() {
    $('.btn').each(function() {
        const btn = $(this);
        if (btn.data('original-text')) {
            btn.html(btn.data('original-text'));
            btn.prop('disabled', false);
        }
    });
}

// Error handling wrapper for API calls
function handleApiCall(promise, successCallback, errorMessage = 'Operation failed') {
    return promise
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            return successCallback(data);
        })
        .catch(error => {
            console.error('API Error:', error);
            showAlert(`${errorMessage}: ${error.message}`, 'danger');
            throw error;
        })
        .finally(() => {
            hideLoading();
            restoreButtonStates();
        });
}

// Keyboard shortcuts
function setupKeyboardShortcuts() {
    $(document).on('keydown', function(e) {
        // Ctrl/Cmd + Enter to run clustering
        if ((e.ctrlKey || e.metaKey) && e.keyCode === 13) {
            e.preventDefault();
            performClustering();
        }

        // Escape to hide loading
        if (e.keyCode === 27) {
            hideLoading();
        }
    });
}

// Data validation helpers
function validateClusteringParams() {
    const nClusters = parseInt($('#n-clusters').val());
    const features = $('#cluster-features').val();
    const method = $('#cluster-method').val();

    const errors = [];

    if (isNaN(nClusters) || nClusters < 2 || nClusters > 10) {
        errors.push('Number of clusters must be between 2 and 10');
    }

    if (!features || features.length === 0) {
        errors.push('Please select at least one feature for clustering');
    }

    if (!method) {
        errors.push('Please select a clustering method');
    }

    return errors;
}

// Performance monitoring
let performanceMetrics = {
    clusteringTime: 0,
    visualizationTime: 0,
    apiCalls: 0
};

function trackPerformance(operation, startTime) {
    const endTime = Date.now();
    const duration = endTime - startTime;
    performanceMetrics[operation + 'Time'] = duration;
    performanceMetrics.apiCalls++;

    console.log(`Performance: ${operation} took ${duration}ms`);

    // Show performance warning if operation is slow
    if (duration > 10000) {
        showAlert(`${operation} took ${(duration/1000).toFixed(1)} seconds. Consider reducing data size.`, 'warning');
    }
}

// Initialize everything when document is ready
$(document).ready(function() {
    initializePage();
    initializeUIEnhancements();
    setupKeyboardShortcuts();

    // Add version info to console
    console.log('%cWorkWell Analytics - Clustering Module', 'color: #667eea; font-size: 16px; font-weight: bold;');
    console.log('Version: 1.0.0');
    console.log('Keyboard shortcuts: Ctrl+Enter (Run clustering), Esc (Cancel loading)');
});

// Export functions for external use
window.ClusteringModule = {
    performClustering,
    analyzeDrivers,
    exportPersonas,
    generateReport,
    showAlert,
    showLoading,
    hideLoading
};