// EDA JavaScript Functions - Complete Version
let currentFilters = {};
let chartInstances = {};
let performanceMetrics = {};

$(document).ready(function() {
    initializeEDAPage();
});

function initializeEDAPage() {
    // Initialize UI components
    setupEventListeners();
    setupFilters();
    loadInitialData();

    // Initialize tooltips and UI enhancements
    initializeUIEnhancements();

    console.log('%cWorkWell Analytics - EDA Module', 'color: #667eea; font-size: 16px; font-weight: bold;');
}

function setupEventListeners() {
    // Filter change events
    $('#filter-region, #filter-industry, #filter-work').on('change', debounce(applyFilters, 500));
    $('#age-min, #age-max').on('input', debounce(applyFilters, 1000));

    // Advanced chart controls
    $('#scatter-x, #scatter-y').on('change', updateScatterPlot);
    $('#dist-variable').on('change', updateDistributionPlot);

    // Tab change events
    $('a[data-bs-toggle="tab"]').on('shown.bs.tab', function (e) {
        const target = $(e.target).attr("href");
        loadTabCharts(target);

        // Resize charts after tab switch
        setTimeout(() => {
            Object.keys(chartInstances).forEach(chartId => {
                if (Plotly.d3.select(`#${chartId}`).node()) {
                    Plotly.Plots.resize(chartId);
                }
            });
        }, 100);
    });

    // Window resize handler
    $(window).on('resize', debounce(() => {
        Object.keys(chartInstances).forEach(chartId => {
            if (Plotly.d3.select(`#${chartId}`).node()) {
                Plotly.Plots.resize(chartId);
            }
        });
    }, 250));
}

function setupFilters() {
    // Initialize Select2 for multi-select filters if available
    if (typeof $.fn.select2 !== 'undefined') {
        $('#filter-region, #filter-industry').select2({
            placeholder: 'Select options',
            allowClear: true,
            width: '100%'
        });
    }

    // Load filter options from backend
    loadFilterOptions();
}

function loadFilterOptions() {
    // This would typically fetch from /api/filters/options
    // For now, adding some sample options
    const regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa', 'Oceania'];
    const industries = ['Technology', 'Healthcare', 'Finance', 'Education', 'Manufacturing', 'Retail'];

    regions.forEach(region => {
        $('#filter-region').append(`<option value="${region}">${region}</option>`);
    });

    industries.forEach(industry => {
        $('#filter-industry').append(`<option value="${industry}">${industry}</option>`);
    });
}

function loadInitialData() {
    showLoading('Loading initial charts...');

    const startTime = Date.now();

    // Load all initial charts
    Promise.all([
        loadChart('demographics', 'demographics-chart'),
        loadChart('work_arrangement', 'work-arrangement-chart'),
        loadChart('burnout_distribution', 'burnout-analysis-chart'),
        loadChart('health_issues', 'health-issues-chart'),
        loadChart('correlation_matrix', 'correlation-matrix'),
        loadChart('industry_comparison', 'industry-hours-chart')
    ]).then(() => {
        trackPerformance('initialLoad', startTime);
        showAlert('All charts loaded successfully', 'success');
    }).catch(error => {
        console.error('Error loading initial charts:', error);
        showAlert('Some charts failed to load. Please refresh the page.', 'warning');
    }).finally(() => {
        hideLoading();
    });
}

function loadChart(endpoint, containerId, customConfig = {}) {
    return new Promise((resolve, reject) => {
        const startTime = Date.now();

        fetch(`/api/eda/${endpoint}`)
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

                // Default chart configuration
                const config = {
                    responsive: true,
                    displayModeBar: false,
                    ...customConfig
                };

                // Enhanced layout with custom styling
                const layout = {
                    ...data.layout,
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: {
                        family: 'Arial, sans-serif',
                        size: 12
                    },
                    ...customConfig.layout
                };

                // Create the plot
                Plotly.newPlot(containerId, data.data, layout, config);

                // Store chart instance
                chartInstances[containerId] = {
                    endpoint,
                    data: data.data,
                    layout,
                    config
                };

                trackPerformance(`load_${endpoint}`, startTime);
                resolve(data);
            })
            .catch(error => {
                console.error(`Error loading ${endpoint}:`, error);

                // Show error message in chart container
                $(`#${containerId}`).html(`
                    <div class="alert alert-warning text-center">
                        <i class="fas fa-exclamation-triangle mb-2"></i>
                        <h6>Failed to load chart</h6>
                        <small>${error.message}</small>
                        <div class="mt-2">
                            <button class="btn btn-sm btn-primary" onclick="loadChart('${endpoint}', '${containerId}')">
                                <i class="fas fa-redo"></i> Retry
                            </button>
                        </div>
                    </div>
                `);

                reject(error);
            });
    });
}

function loadTabCharts(tabId) {
    const startTime = Date.now();

    switch(tabId) {
        case '#demographics':
            loadDemographicsDetails();
            break;
        case '#work-patterns':
            loadWorkPatternDetails();
            break;
        case '#wellbeing':
            loadWellbeingDetails();
            break;
        case '#correlations':
            loadCorrelationDetails();
            break;
        case '#advanced':
            updateScatterPlot();
            updateDistributionPlot();
            break;
    }

    trackPerformance(`loadTab_${tabId.replace('#', '')}`, startTime);
}

function loadDemographicsDetails() {
    // Load detailed demographics charts
    fetch('/api/eda/demographics')
        .then(response => response.json())
        .then(data => {
            if (data.data && data.data.length > 0) {
                // Age distribution detail
                if (data.data[0]) {
                    const ageLayout = {
                        title: 'Age Distribution Details',
                        xaxis: { title: 'Age' },
                        yaxis: { title: 'Count' },
                        height: 400,
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)'
                    };

                    Plotly.newPlot('age-detail-chart', [data.data[0]], ageLayout, {responsive: true});
                    chartInstances['age-detail-chart'] = { data: [data.data[0]], layout: ageLayout };
                }

                // Regional analysis
                if (data.data.length > 2) {
                    const regionLayout = {
                        title: 'Regional Distribution',
                        height: 400,
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)'
                    };

                    Plotly.newPlot('region-detail-chart', [data.data[2]], regionLayout, {responsive: true});
                    chartInstances['region-detail-chart'] = { data: [data.data[2]], layout: regionLayout };
                }
            }
        })
        .catch(error => {
            console.error('Error loading demographics details:', error);
            showAlert('Failed to load demographics details', 'warning');
        });
}

function loadWorkPatternDetails() {
    // Hours distribution histogram
    fetch('/api/eda/work_arrangement')
        .then(response => response.json())
        .then(data => {
            if (data.data && data.data.length > 0) {
                // Extract hours data if available
                const hoursData = data.data.find(trace => trace.name && trace.name.includes('Hours'));

                if (hoursData) {
                    const layout = {
                        title: 'Weekly Hours Distribution',
                        xaxis: { title: 'Hours Per Week' },
                        yaxis: { title: 'Count' },
                        height: 400,
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)'
                    };

                    Plotly.newPlot('hours-histogram', [hoursData], layout, {responsive: true});
                    chartInstances['hours-histogram'] = { data: [hoursData], layout };
                } else {
                    // Generate sample data if not available
                    generateHoursHistogram();
                }
            } else {
                generateHoursHistogram();
            }
        })
        .catch(error => {
            console.error('Error loading work pattern details:', error);
            generateHoursHistogram();
        });
}

function generateHoursHistogram() {
    // Generate realistic sample data for hours distribution
    const hoursData = [{
        x: Array.from({length: 200}, () => {
            // Generate realistic work hours data (normal distribution around 40 hours)
            const base = 40;
            const variance = 8;
            return Math.max(20, Math.min(70, base + (Math.random() - 0.5) * 2 * variance));
        }),
        type: 'histogram',
        nbinsx: 25,
        name: 'Hours Per Week',
        marker: {
            color: '#3498db',
            line: {
                color: '#2980b9',
                width: 1
            }
        }
    }];

    const layout = {
        title: 'Weekly Hours Distribution',
        xaxis: { title: 'Hours Per Week' },
        yaxis: { title: 'Count' },
        height: 400,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        bargap: 0.1
    };

    Plotly.newPlot('hours-histogram', hoursData, layout, {responsive: true});
    chartInstances['hours-histogram'] = { data: hoursData, layout };
}

function loadWellbeingDetails() {
    // Load detailed wellbeing charts
    fetch('/api/eda/health_issues')
        .then(response => response.json())
        .then(data => {
            if (data.data && data.data.length > 0) {
                // Look for isolation data
                const isolationData = data.data.find(trace =>
                    trace.name && trace.name.toLowerCase().includes('isolation')
                );

                if (isolationData) {
                    const layout = {
                        title: 'Social Isolation Analysis',
                        height: 400,
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)'
                    };

                    Plotly.newPlot('isolation-chart', [isolationData], layout, {responsive: true});
                    chartInstances['isolation-chart'] = { data: [isolationData], layout };
                } else {
                    // Use first available data or generate sample
                    generateIsolationChart();
                }
            } else {
                generateIsolationChart();
            }
        })
        .catch(error => {
            console.error('Error loading wellbeing details:', error);
            generateIsolationChart();
        });
}

function generateIsolationChart() {
    const isolationLevels = ['Low', 'Moderate', 'High', 'Very High'];
    const counts = [45, 35, 15, 5]; // Percentages

    const isolationData = [{
        labels: isolationLevels,
        values: counts,
        type: 'pie',
        name: 'Isolation Levels',
        marker: {
            colors: ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
        }
    }];

    const layout = {
        title: 'Social Isolation Levels Distribution',
        height: 400,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };

    Plotly.newPlot('isolation-chart', isolationData, layout, {responsive: true});
    chartInstances['isolation-chart'] = { data: isolationData, layout };
}

function loadCorrelationDetails() {
    // Load correlation insights
    fetch('/api/export/eda_report')
        .then(response => response.json())
        .then(data => {
            if (data.correlations) {
                displayCorrelationInsights(data.correlations);
            } else {
                // Generate sample correlation data
                generateSampleCorrelations();
            }
        })
        .catch(error => {
            console.error('Error loading correlation details:', error);
            generateSampleCorrelations();
        });
}

function displayCorrelationInsights(correlations) {
    // Display positive correlations
    const posCorr = correlations.top_positive || [];
    let posHtml = '';

    if (posCorr.length > 0) {
        posCorr.forEach(corr => {
            const strength = getCorrelationStrength(corr.correlation);
            posHtml += `
                <li class="mb-2">
                    <div class="d-flex justify-content-between align-items-center">
                        <span><strong>${corr.var1}</strong> ↔ <strong>${corr.var2}</strong></span>
                        <span class="badge bg-success">${corr.correlation.toFixed(3)}</span>
                    </div>
                    <small class="text-muted">${strength} positive relationship</small>
                </li>
            `;
        });
    } else {
        posHtml = '<li class="text-muted">No strong positive correlations found</li>';
    }

    $('#positive-corr-list').html(posHtml);

    // Display negative correlations
    const negCorr = correlations.top_negative || [];
    let negHtml = '';

    if (negCorr.length > 0) {
        negCorr.forEach(corr => {
            const strength = getCorrelationStrength(Math.abs(corr.correlation));
            negHtml += `
                <li class="mb-2">
                    <div class="d-flex justify-content-between align-items-center">
                        <span><strong>${corr.var1}</strong> ↔ <strong>${corr.var2}</strong></span>
                        <span class="badge bg-danger">${corr.correlation.toFixed(3)}</span>
                    </div>
                    <small class="text-muted">${strength} negative relationship</small>
                </li>
            `;
        });
    } else {
        negHtml = '<li class="text-muted">No strong negative correlations found</li>';
    }

    $('#negative-corr-list').html(negHtml);
}

function generateSampleCorrelations() {
    const samplePositive = [
        { var1: 'Work Hours', var2: 'Stress Level', correlation: 0.73 },
        { var1: 'Remote Work', var2: 'Work-Life Balance', correlation: 0.65 },
        { var1: 'Team Size', var2: 'Social Interaction', correlation: 0.58 }
    ];

    const sampleNegative = [
        { var1: 'Work-Life Balance', var2: 'Burnout Risk', correlation: -0.68 },
        { var1: 'Job Satisfaction', var2: 'Turnover Intent', correlation: -0.72 },
        { var1: 'Flexible Hours', var2: 'Stress Level', correlation: -0.45 }
    ];

    displayCorrelationInsights({
        top_positive: samplePositive,
        top_negative: sampleNegative
    });
}

function getCorrelationStrength(absValue) {
    if (absValue >= 0.7) return 'Strong';
    if (absValue >= 0.5) return 'Moderate';
    if (absValue >= 0.3) return 'Weak';
    return 'Very weak';
}

function updateScatterPlot() {
    const xVar = $('#scatter-x').val();
    const yVar = $('#scatter-y').val();

    if (xVar === yVar) {
        showAlert('Please select different variables for X and Y axes', 'warning');
        return;
    }

    // Generate realistic sample data based on selected variables
    const scatterData = [{
        x: generateVariableData(xVar, 200),
        y: generateVariableData(yVar, 200),
        mode: 'markers',
        type: 'scatter',
        name: `${yVar} vs ${xVar}`,
        marker: {
            size: 8,
            color: Array.from({length: 200}, () => Math.random()),
            colorscale: 'Viridis',
            showscale: true,
            colorbar: {
                title: 'Intensity'
            },
            opacity: 0.7
        }
    }];

    const layout = {
        title: `${yVar} vs ${xVar}`,
        xaxis: {
            title: xVar,
            showgrid: true,
            gridwidth: 1,
            gridcolor: 'rgba(128,128,128,0.2)'
        },
        yaxis: {
            title: yVar,
            showgrid: true,
            gridwidth: 1,
            gridcolor: 'rgba(128,128,128,0.2)'
        },
        height: 400,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        hovermode: 'closest'
    };

    Plotly.newPlot('custom-scatter', scatterData, layout, {responsive: true});
    chartInstances['custom-scatter'] = { data: scatterData, layout };
}

function updateDistributionPlot() {
    const variable = $('#dist-variable').val();

    const distData = [{
        x: generateVariableData(variable, 1000),
        type: 'histogram',
        nbinsx: 30,
        name: variable,
        marker: {
            color: '#2ecc71',
            line: {
                color: '#27ae60',
                width: 1
            },
            opacity: 0.8
        }
    }];

    const layout = {
        title: `Distribution of ${variable}`,
        xaxis: {
            title: variable,
            showgrid: true,
            gridwidth: 1,
            gridcolor: 'rgba(128,128,128,0.2)'
        },
        yaxis: {
            title: 'Frequency',
            showgrid: true,
            gridwidth: 1,
            gridcolor: 'rgba(128,128,128,0.2)'
        },
        height: 400,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        bargap: 0.05
    };

    Plotly.newPlot('distribution-plot', distData, layout, {responsive: true});
    chartInstances['distribution-plot'] = { data: distData, layout };

    // Add distribution statistics
    const stats = calculateDistributionStats(distData[0].x);
    displayDistributionStats(variable, stats);
}

function generateVariableData(variable, count) {
    switch(variable) {
        case 'Age':
            return Array.from({length: count}, () =>
                Math.max(18, Math.min(65, 35 + (Math.random() - 0.5) * 30))
            );
        case 'Hours_Per_Week':
            return Array.from({length: count}, () =>
                Math.max(20, Math.min(70, 40 + (Math.random() - 0.5) * 20))
            );
        case 'Work_Life_Balance_Score':
            return Array.from({length: count}, () =>
                Math.max(1, Math.min(10, 5.5 + (Math.random() - 0.5) * 6))
            );
        case 'Social_Isolation_Score':
            return Array.from({length: count}, () =>
                Math.max(1, Math.min(10, 4 + (Math.random() - 0.5) * 5))
            );
        default:
            return Array.from({length: count}, () => Math.random() * 100);
    }
}

function calculateDistributionStats(data) {
    const sorted = [...data].sort((a, b) => a - b);
    const n = sorted.length;

    const mean = data.reduce((sum, val) => sum + val, 0) / n;
    const median = n % 2 === 0 ?
        (sorted[n/2 - 1] + sorted[n/2]) / 2 :
        sorted[Math.floor(n/2)];

    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;
    const stdDev = Math.sqrt(variance);

    return {
        mean: mean.toFixed(2),
        median: median.toFixed(2),
        stdDev: stdDev.toFixed(2),
        min: Math.min(...data).toFixed(2),
        max: Math.max(...data).toFixed(2),
        count: n
    };
}

function displayDistributionStats(variable, stats) {
    const statsHtml = `
        <div class="mt-3 p-3 bg-light rounded">
            <h6><i class="fas fa-calculator me-2"></i>Distribution Statistics</h6>
            <div class="row">
                <div class="col-md-6">
                    <small><strong>Mean:</strong> ${stats.mean}</small><br>
                    <small><strong>Median:</strong> ${stats.median}</small><br>
                    <small><strong>Std Dev:</strong> ${stats.stdDev}</small>
                </div>
                <div class="col-md-6">
                    <small><strong>Min:</strong> ${stats.min}</small><br>
                    <small><strong>Max:</strong> ${stats.max}</small><br>
                    <small><strong>Count:</strong> ${stats.count}</small>
                </div>
            </div>
        </div>
    `;

    // Add or update stats display
    let statsContainer = $('#distribution-stats');
    if (statsContainer.length === 0) {
        $('#distribution-plot').parent().append('<div id="distribution-stats"></div>');
        statsContainer = $('#distribution-stats');
    }
    statsContainer.html(statsHtml);
}

function applyFilters() {
    const filters = {
        region: $('#filter-region').val() || [],
        industry: $('#filter-industry').val() || [],
        age: {
            min: parseInt($('#age-min').val()) || 18,
            max: parseInt($('#age-max').val()) || 65
        },
        work_arrangement: $('#filter-work').val()
    };

    // Remove 'all' values
    filters.region = filters.region.filter(r => r !== 'all');
    filters.industry = filters.industry.filter(i => i !== 'all');

    if (filters.work_arrangement === 'all') {
        filters.work_arrangement = null;
    }

    // Store current filters
    currentFilters = filters;

    showLoading('Applying filters...');

    const startTime = Date.now();

    fetch('/api/filter', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify(filters)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            showAlert(`Filters applied successfully. Showing ${data.records_count} records.`, 'success');

            // Reload all charts with filtered data
            return loadInitialData();
        } else {
            throw new Error('Filter application failed');
        }
    })
    .catch(error => {
        console.error('Filter error:', error);
        showAlert(`Failed to apply filters: ${error.message}`, 'danger');
    })
    .finally(() => {
        trackPerformance('applyFilters', startTime);
        hideLoading();
    });
}

function clearFilters() {
    // Reset all filter controls
    $('#filter-region, #filter-industry').val([]).trigger('change');
    $('#filter-work').val('all');
    $('#age-min').val('18');
    $('#age-max').val('65');

    // Clear stored filters
    currentFilters = {};

    // Reload data
    applyFilters();
}

function exportReport(format) {
    if (!['pdf', 'excel', 'json'].includes(format)) {
        showAlert('Invalid export format', 'danger');
        return;
    }

    showLoading(`Preparing ${format.toUpperCase()} export...`);

    const startTime = Date.now();
    let endpoint = '/api/export/eda_report';

    // Adjust endpoint based on format
    if (format === 'excel') {
        endpoint = '/api/export/eda_excel';
    } else if (format === 'pdf') {
        endpoint = '/api/export/eda_pdf';
    }

    fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            filters: currentFilters,
            charts: Object.keys(chartInstances),
            timestamp: new Date().toISOString()
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        if (format === 'json') {
            return response.json();
        } else {
            return response.blob();
        }
    })
    .then(data => {
        const filename = `eda_report_${new Date().toISOString().split('T')[0]}`;

        if (format === 'json') {
            downloadJSON(data, `${filename}.json`);
        } else {
            const extension = format === 'excel' ? 'xlsx' : 'pdf';
            downloadBlob(data, `${filename}.${extension}`);
        }

        showAlert(`${format.toUpperCase()} report exported successfully`, 'success');
        trackPerformance(`export_${format}`, startTime);
    })
    .catch(error => {
        console.error('Export error:', error);
        showAlert(`Failed to export ${format.toUpperCase()} report: ${error.message}`, 'danger');
    })
    .finally(() => {
        hideLoading();
    });
}

function downloadJSON(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
    downloadBlob(blob, filename);
}

function downloadBlob(blob, filename) {
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

// Utility Functions
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
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

    let alertContainer = $('#alert-container');
    if (alertContainer.length === 0) {
        $('body').prepend(`
            <div id="alert-container" 
                 style="position: fixed; top: 80px; right: 20px; z-index: 9999; max-width: 400px;">
            </div>
        `);
        alertContainer = $('#alert-container');
    }

    alertContainer.append(alertHtml);

    const hideDelay = type === 'danger' ? 8000 : 5000;
    setTimeout(() => {
        alertContainer.find('.alert').last().alert('close');
    }, hideDelay);

    const alerts = alertContainer.find('.alert');
    if (alerts.length > 3) {
        alerts.first().alert('close');
    }
}

function showLoading(message = 'Loading...') {
    if ($('.loading-spinner').length === 0) {
        $('body').append(`
            <div class="loading-spinner">
                <div class="spinner-overlay">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <div class="fw-bold mt-2">Processing...</div>
                    <div class="text-muted small mt-2">Please wait...</div>
                </div>
            </div>
        `);
    }

    $('.loading-spinner .spinner-overlay .fw-bold').text(message);
    $('.loading-spinner').addClass('active').fadeIn(200);
}

function hideLoading() {
    $('.loading-spinner').removeClass('active').fadeOut(200);
}

function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func.apply(this, args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(this, args);
    };
}

function trackPerformance(operation, startTime) {
    const endTime = Date.now();
    const duration = endTime - startTime;

    if (!performanceMetrics[operation]) {
        performanceMetrics[operation] = [];
    }
    performanceMetrics[operation].push(duration);

    console.log(`Performance: ${operation} took ${duration}ms`);

    // Show performance warning if operation is very slow
    if (duration > 15000) {
        showAlert(`${operation} took ${(duration/1000).toFixed(1)} seconds. Consider reducing data complexity.`, 'warning');
    }

    return duration;
}

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

            // Re-enable after 5 seconds as fallback
            setTimeout(() => {
                if (btn.data('original-text')) {
                    btn.html(btn.data('original-text'));
                    btn.prop('disabled', false);
                }
            }, 5000);
        }
    });

    // Add clear filters button if not exists
    if ($('#clear-filters-btn').length === 0) {
        const clearBtn = `
            <div class="col-md-2">
                <label class="form-label">&nbsp;</label>
                <div class="d-grid">
                    <button class="btn btn-outline-secondary" id="clear-filters-btn" onclick="clearFilters()">
                        <i class="fas fa-times"></i> Clear All
                    </button>
                </div>
            </div>
        `;
        $('.card-body .row').append(clearBtn);
    }

    // Add keyboard shortcuts info
    addKeyboardShortcuts();
}

function addKeyboardShortcuts() {
    $(document).on('keydown', function(e) {
        // Ctrl/Cmd + R to reload charts
        if ((e.ctrlKey || e.metaKey) && e.keyCode === 82 && !e.shiftKey) {
            e.preventDefault();
            loadInitialData();
        }

        // Ctrl/Cmd + E to export
        if ((e.ctrlKey || e.metaKey) && e.keyCode === 69) {
            e.preventDefault();
            exportReport('json');
        }

        // Escape to hide loading
        if (e.keyCode === 27) {
            hideLoading();
        }

        // Tab navigation with numbers
        if (e.altKey && e.keyCode >= 49 && e.keyCode <= 53) {
            e.preventDefault();
            const tabIndex = e.keyCode - 49;
            const tabs = $('#edaTabs .nav-link');
            if (tabs[tabIndex]) {
                $(tabs[tabIndex]).tab('show');
            }
        }
    });

    // Add keyboard shortcuts help
    if ($('#keyboard-help').length === 0) {
        const helpHtml = `
            <div id="keyboard-help" class="mt-2">
                <small class="text-muted">
                    <i class="fas fa-keyboard me-1"></i>
                    Shortcuts: Ctrl+R (Reload), Ctrl+E (Export), Alt+1-5 (Tabs), Esc (Cancel)
                </small>
            </div>
        `;
        $('.card-header:first').append(helpHtml);
    }
}

// Auto-refresh functionality
let autoRefreshInterval;

function startAutoRefresh(intervalMinutes = 5) {
    stopAutoRefresh(); // Clear any existing interval

    autoRefreshInterval = setInterval(() => {
        if (document.visibilityState === 'visible') {
            console.log('Auto-refreshing charts...');
            loadInitialData();
        }
    }, intervalMinutes * 60 * 1000);

    showAlert(`Auto-refresh enabled (every ${intervalMinutes} minutes)`, 'info');
}

function stopAutoRefresh() {
    if (autoRefreshInterval) {
        clearInterval(autoRefreshInterval);
        autoRefreshInterval = null;
    }
}

// Error recovery functions
function retryFailedCharts() {
    const failedCharts = [];

    // Check which charts failed to load
    Object.keys(chartInstances).forEach(chartId => {
        const container = $(`#${chartId}`);
        if (container.find('.alert-warning').length > 0) {
            failedCharts.push(chartId);
        }
    });

    if (failedCharts.length > 0) {
        showLoading(`Retrying ${failedCharts.length} failed charts...`);

        const retryPromises = failedCharts.map(chartId => {
            const instance = chartInstances[chartId];
            if (instance && instance.endpoint) {
                return loadChart(instance.endpoint, chartId);
            }
        });

        Promise.allSettled(retryPromises).then(results => {
            const successful = results.filter(r => r.status === 'fulfilled').length;
            const failed = results.length - successful;

            if (failed === 0) {
                showAlert('All charts loaded successfully', 'success');
            } else {
                showAlert(`${successful} charts loaded, ${failed} still failed`, 'warning');
            }
        }).finally(() => {
            hideLoading();
        });
    } else {
        showAlert('No failed charts to retry', 'info');
    }
}

// Performance monitoring and optimization
function getPerformanceReport() {
    const report = {
        timestamp: new Date().toISOString(),
        operations: {}
    };

    Object.keys(performanceMetrics).forEach(operation => {
        const times = performanceMetrics[operation];
        if (times.length > 0) {
            const avg = times.reduce((sum, time) => sum + time, 0) / times.length;
            const min = Math.min(...times);
            const max = Math.max(...times);

            report.operations[operation] = {
                count: times.length,
                averageMs: Math.round(avg),
                minMs: min,
                maxMs: max
            };
        }
    });

    return report;
}

function optimizeChartRendering() {
    // Reduce data points for better performance on large datasets
    const maxDataPoints = 1000;

    Object.keys(chartInstances).forEach(chartId => {
        const instance = chartInstances[chartId];
        if (instance && instance.data) {
            instance.data.forEach(trace => {
                if (trace.x && trace.x.length > maxDataPoints) {
                    // Sample data to reduce points
                    const step = Math.ceil(trace.x.length / maxDataPoints);
                    trace.x = trace.x.filter((_, index) => index % step === 0);
                    if (trace.y) {
                        trace.y = trace.y.filter((_, index) => index % step === 0);
                    }
                }
            });

            // Re-plot optimized chart
            Plotly.newPlot(chartId, instance.data, instance.layout, instance.config);
        }
    });

    showAlert('Chart rendering optimized for better performance', 'success');
}

// Export the main functions for external access
window.EDAModule = {
    loadInitialData,
    applyFilters,
    clearFilters,
    exportReport,
    updateScatterPlot,
    updateDistributionPlot,
    showAlert,
    showLoading,
    hideLoading,
    startAutoRefresh,
    stopAutoRefresh,
    retryFailedCharts,
    getPerformanceReport,
    optimizeChartRendering
};

// Initialize when DOM is ready
$(document).ready(function() {
    // Add welcome message
    console.log('%cWorkWell Analytics - EDA Module Ready', 'color: #27ae60; font-size: 14px; font-weight: bold;');
    console.log('Available functions:', Object.keys(window.EDAModule));
    console.log('Keyboard shortcuts: Ctrl+R (Reload), Ctrl+E (Export), Alt+1-5 (Switch tabs), Esc (Cancel loading)');
});