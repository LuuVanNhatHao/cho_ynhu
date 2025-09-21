// Hypothesis Testing JavaScript - Complete Version
let testResults = {};
let significantCount = 0;
let testRunning = false;

$(document).ready(function() {
    initializeHypothesisPage();
});

function initializeHypothesisPage() {
    // Load pre-computed results if available
    loadInitialResults();

    // Setup UI enhancements
    initializeUIEnhancements();

    // Setup keyboard shortcuts
    setupKeyboardShortcuts();

    console.log('%cWorkWell Analytics - Hypothesis Testing Module', 'color: #667eea; font-size: 16px; font-weight: bold;');
}

function runTest(testType) {
    if (testRunning) {
        showAlert('Another test is currently running. Please wait.', 'warning');
        return;
    }

    testRunning = true;
    showLoading(`Running ${getTestName(testType)}...`);

    // Update button state
    updateButtonState(testType, 'running');

    const startTime = Date.now();

    fetch(`/api/hypothesis/${testType}`)
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

            // Store result
            testResults[testType] = data.result;

            // Update UI based on test type
            updateTestUI(testType, data.result);

            // Display chart if available
            if (data.chart) {
                const chartId = getChartId(testType);
                displayChart(chartId, data.chart);
            }

            // Update summary
            updateSummary();

            // Track performance
            const duration = Date.now() - startTime;
            console.log(`Test ${testType} completed in ${duration}ms`);

            showAlert(`${getTestName(testType)} completed successfully`, 'success');
        })
        .catch(error => {
            console.error(`Error running ${testType}:`, error);
            showAlert(`Failed to run ${getTestName(testType)}: ${error.message}`, 'danger');

            // Reset button state on error
            updateButtonState(testType, 'error');
        })
        .finally(() => {
            hideLoading();
            testRunning = false;
            restoreButtonState(testType);
        });
}

function displayChart(chartId, chartData) {
    try {
        const config = {
            responsive: true,
            displayModeBar: false,
            showTips: true
        };

        const layout = {
            ...chartData.layout,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {
                family: 'Arial, sans-serif',
                size: 12
            }
        };

        Plotly.newPlot(chartId, chartData.data, layout, config);

        // Add click handlers for interactivity
        const chartElement = document.getElementById(chartId);
        if (chartElement) {
            chartElement.on('plotly_click', function(data) {
                if (data.points && data.points.length > 0) {
                    const point = data.points[0];
                    showAlert(`Data point: ${point.x}, ${point.y}`, 'info');
                }
            });
        }
    } catch (error) {
        console.error(`Error displaying chart ${chartId}:`, error);
        $(`#${chartId}`).html(`
            <div class="alert alert-warning text-center">
                <i class="fas fa-exclamation-triangle mb-2"></i>
                <h6>Chart Display Error</h6>
                <small>${error.message}</small>
            </div>
        `);
    }
}

function updateTestUI(testType, result) {
    try {
        switch(testType) {
            case 'remote_burnout':
                updateRemoteBurnoutUI(result);
                break;
            case 'gender_balance':
                updateGenderBalanceUI(result);
                break;
            case 'age_mental_health':
                updateAgeMentalHealthUI(result);
                break;
            case 'industry_hours':
                updateIndustryHoursUI(result);
                break;
            case 'salary_isolation':
                updateSalaryIsolationUI(result);
                break;
            default:
                console.warn(`Unknown test type: ${testType}`);
        }
    } catch (error) {
        console.error(`Error updating UI for ${testType}:`, error);
        showAlert(`Failed to update results for ${getTestName(testType)}`, 'warning');
    }
}

function updateRemoteBurnoutUI(result) {
    $('#test1-samples').text(`Remote: ${result.sample_sizes?.remote || 'N/A'}, Office: ${result.sample_sizes?.office || 'N/A'}`);
    $('#test1-statistic').text((result.statistic || 0).toFixed(2));
    $('#test1-pvalue').text((result.p_value || 0).toFixed(4));
    $('#test1-effect').text(`${(result.effect_size || 0).toFixed(3)} (${interpretEffectSize(result.effect_size || 0)})`);

    const significant = result.significant || false;
    updateResultBadge('test1-result', significant);

    const interpretation = `
        <strong>${result.conclusion || 'Analysis completed'}</strong><br>
        Remote workers show a mean burnout score of ${(result.remote_mean || 0).toFixed(2)} 
        compared to ${(result.office_mean || 0).toFixed(2)} for office workers. 
        The difference is ${significant ? 'statistically significant' : 'not statistically significant'} 
        (p = ${(result.p_value || 0).toFixed(4)}) with ${interpretEffectSize(result.effect_size || 0)} effect size.
    `;
    $('#test1-interpretation').html(interpretation);
}

function updateGenderBalanceUI(result) {
    $('#test2-male').text(`${(result.male_mean || 0).toFixed(2)} ± ${(result.male_std || 0).toFixed(2)}`);
    $('#test2-female').text(`${(result.female_mean || 0).toFixed(2)} ± ${(result.female_std || 0).toFixed(2)}`);
    $('#test2-pvalue').text((result.p_value || 0).toFixed(4));
    $('#test2-effect').text(`${(result.effect_size || 0).toFixed(3)} (${interpretEffectSize(Math.abs(result.effect_size || 0))})`);

    const significant = result.significant || false;
    updateResultBadge('test2-result', significant);

    const interpretation = `
        <strong>${result.conclusion || 'Analysis completed'}</strong><br>
        Males have a mean WLB score of ${(result.male_mean || 0).toFixed(2)} while females have ${(result.female_mean || 0).toFixed(2)}. 
        Cohen's d = ${Math.abs(result.effect_size || 0).toFixed(3)} indicates ${interpretEffectSize(Math.abs(result.effect_size || 0))} effect. 
        ${significant ? 'Gender significantly affects' : 'Gender does not significantly affect'} work-life balance scores.
    `;
    $('#test2-interpretation').html(interpretation);
}

function updateAgeMentalHealthUI(result) {
    $('#test3-chi2').text((result.chi2_statistic || 0).toFixed(2));
    $('#test3-pvalue').text((result.chi2_p_value || result.p_value || 0).toFixed(4));
    $('#test3-cramers').text((result.cramers_v || 0).toFixed(3));
    $('#test3-fstat').text((result.f_statistic || 0).toFixed(2));

    const significant = result.significant || false;
    updateResultBadge('test3-result', significant);

    const interpretation = `
        <strong>${result.conclusion || 'Analysis completed'}</strong><br>
        Chi-square test reveals ${significant ? 'significant' : 'no significant'} association between age groups and mental health 
        (χ² = ${(result.chi2_statistic || 0).toFixed(2)}, p = ${(result.chi2_p_value || result.p_value || 0).toFixed(4)}). 
        Cramér's V = ${(result.cramers_v || 0).toFixed(3)} indicates ${interpretCramersV(result.cramers_v || 0)} association strength.
    `;
    $('#test3-interpretation').html(interpretation);
}

function updateIndustryHoursUI(result) {
    const industries = Object.keys(result.industry_stats || {});
    $('#test4-statistic').text((result.statistic || 0).toFixed(2));
    $('#test4-pvalue').text((result.p_value || 0).toFixed(4));
    $('#test4-eta').text(`${(result.eta_squared || 0).toFixed(3)} (${interpretEffectSize(result.eta_squared || 0)})`);
    $('#test4-count').text(industries.length);

    const significant = result.significant || false;
    updateResultBadge('test4-result', significant);

    // Find industry with highest and lowest hours
    let maxHours = 0, minHours = 100, maxInd = 'Unknown', minInd = 'Unknown';
    if (result.industry_stats && Object.keys(result.industry_stats).length > 0) {
        for (let ind in result.industry_stats) {
            const mean = result.industry_stats[ind].mean || 0;
            if (mean > maxHours) { maxHours = mean; maxInd = ind; }
            if (mean < minHours) { minHours = mean; minInd = ind; }
        }
    }

    const interpretation = `
        <strong>${result.conclusion || 'Analysis completed'}</strong><br>
        Kruskal-Wallis test shows ${significant ? 'significant' : 'no significant'} differences in working hours across industries 
        (H = ${(result.statistic || 0).toFixed(2)}, p = ${(result.p_value || 0).toFixed(4)}). 
        ${maxInd} has the highest average (${maxHours.toFixed(1)} hrs/week) while ${minInd} has the lowest (${minHours.toFixed(1)} hrs/week).
    `;
    $('#test4-interpretation').html(interpretation);
}

function updateSalaryIsolationUI(result) {
    $('#test5-correlation').text((result.correlation || 0).toFixed(3));
    $('#test5-pvalue').text((result.p_value || 0).toFixed(4));
    $('#test5-direction').text(result.direction || 'unknown');
    $('#test5-strength').text(result.strength || 'unknown');

    const significant = result.significant || false;
    updateResultBadge('test5-result', significant);

    const interpretation = `
        <strong>${result.conclusion || 'Analysis completed'}</strong><br>
        Spearman correlation reveals ${result.strength || 'unknown'} ${result.direction || 'unknown'} correlation 
        between salary and social isolation (ρ = ${(result.correlation || 0).toFixed(3)}, p = ${(result.p_value || 0).toFixed(4)}). 
        ${(result.direction === 'positive') ? 'Higher salaries are associated with increased' : 'Higher salaries are associated with decreased'} social isolation scores.
    `;
    $('#test5-interpretation').html(interpretation);
}

function updateResultBadge(elementId, significant) {
    const element = $(`#${elementId}`);
    element.removeClass('bg-secondary bg-success bg-danger bg-warning');

    if (significant === true) {
        element.addClass('bg-success').text('Significant');
    } else if (significant === false) {
        element.addClass('bg-danger').text('Not Significant');
    } else {
        element.addClass('bg-warning').text('Inconclusive');
    }
}

function updateButtonState(testType, state) {
    const button = $(`button[onclick*="${testType}"]`);
    const originalText = button.html();

    button.data('original-text', originalText);

    switch(state) {
        case 'running':
            button.html('<i class="fas fa-spinner fa-spin"></i> Running...').prop('disabled', true);
            break;
        case 'error':
            button.html('<i class="fas fa-exclamation-triangle"></i> Error').addClass('btn-danger');
            break;
    }
}

function restoreButtonState(testType) {
    const button = $(`button[onclick*="${testType}"]`);
    const originalText = button.data('original-text');

    if (originalText) {
        button.html(originalText).prop('disabled', false).removeClass('btn-danger');
    }
}

function interpretEffectSize(value) {
    const absValue = Math.abs(value || 0);
    if (absValue < 0.2) return 'negligible';
    if (absValue < 0.5) return 'small';
    if (absValue < 0.8) return 'medium';
    return 'large';
}

function interpretCramersV(value) {
    const absValue = Math.abs(value || 0);
    if (absValue < 0.1) return 'negligible';
    if (absValue < 0.3) return 'weak';
    if (absValue < 0.5) return 'moderate';
    return 'strong';
}

function getChartId(testType) {
    const chartMap = {
        'remote_burnout': 'test1-chart',
        'gender_balance': 'test2-chart',
        'age_mental_health': 'test3-chart',
        'industry_hours': 'test4-chart',
        'salary_isolation': 'test5-chart'
    };
    return chartMap[testType] || 'unknown-chart';
}

function getTestName(testType) {
    const nameMap = {
        'remote_burnout': 'Remote vs Burnout Test',
        'gender_balance': 'Gender vs Work-Life Balance Test',
        'age_mental_health': 'Age vs Mental Health Test',
        'industry_hours': 'Industry vs Working Hours Test',
        'salary_isolation': 'Salary vs Isolation Test'
    };
    return nameMap[testType] || 'Unknown Test';
}

function runAllTests() {
    if (testRunning) {
        showAlert('A test is currently running. Please wait.', 'warning');
        return;
    }

    const tests = ['remote_burnout', 'gender_balance', 'age_mental_health', 'industry_hours', 'salary_isolation'];
    let completedTests = 0;

    showAlert('Running all hypothesis tests...', 'info');
    showLoading('Running comprehensive statistical analysis...');

    // Update progress indicator
    updateProgress(0, tests.length);

    // Run tests sequentially to avoid overwhelming the server
    function runNextTest(index) {
        if (index >= tests.length) {
            hideLoading();
            showAlert(`All tests completed! ${significantCount} significant results found.`, 'success');
            return;
        }

        const testType = tests[index];
        const startTime = Date.now();

        fetch(`/api/hypothesis/${testType}`)
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

                testResults[testType] = data.result;
                updateTestUI(testType, data.result);

                if (data.chart) {
                    const chartId = getChartId(testType);
                    displayChart(chartId, data.chart);
                }

                completedTests++;
                updateProgress(completedTests, tests.length);

                const duration = Date.now() - startTime;
                console.log(`Test ${testType} completed in ${duration}ms`);

                // Run next test after a brief delay
                setTimeout(() => runNextTest(index + 1), 500);
            })
            .catch(error => {
                console.error(`Error running ${testType}:`, error);
                showAlert(`Test ${getTestName(testType)} failed: ${error.message}`, 'warning');

                completedTests++;
                updateProgress(completedTests, tests.length);

                // Continue with next test even if this one failed
                setTimeout(() => runNextTest(index + 1), 500);
            });
    }

    runNextTest(0);
}

function updateProgress(completed, total) {
    const percentage = (completed / total) * 100;
    const progressText = `${completed}/${total} tests completed`;

    // Update loading message
    $('.loading-spinner .spinner-overlay .fw-bold').text(`Running Tests... (${completed}/${total})`);

    // Update summary count as tests complete
    updateSummary();
}

function updateSummary() {
    // Count significant results
    significantCount = 0;
    const findings = [];
    const testCount = Object.keys(testResults).length;

    for (let test in testResults) {
        if (testResults[test] && testResults[test].significant) {
            significantCount++;
            findings.push(testResults[test].conclusion || 'Significant result found');
        }
    }

    // Update significant count with animation
    animateCounterTo('#significant-count', significantCount);

    // Update key findings
    if (findings.length > 0) {
        let findingsHtml = '';
        findings.forEach((finding, index) => {
            findingsHtml += `<li class="mb-2">
                <i class="fas fa-check-circle text-success me-2"></i>
                ${finding}
            </li>`;
        });
        $('#key-findings-list').html(findingsHtml);
    } else if (testCount > 0) {
        $('#key-findings-list').html('<li class="text-muted">No significant findings yet...</li>');
    }

    // Update overall statistics if we have results
    if (testCount > 0) {
        const significanceRate = (significantCount / testCount * 100).toFixed(1);
        updateStatisticalPower(significanceRate);
    }
}

function animateCounterTo(selector, targetValue) {
    const element = $(selector);
    const currentValue = parseInt(element.text()) || 0;

    if (currentValue === targetValue) return;

    const duration = 1000;
    const steps = 20;
    const increment = (targetValue - currentValue) / steps;

    let current = currentValue;
    let step = 0;

    const timer = setInterval(() => {
        step++;
        current += increment;

        if (step >= steps) {
            element.text(targetValue);
            clearInterval(timer);
        } else {
            element.text(Math.round(current));
        }
    }, duration / steps);
}

function updateStatisticalPower(significanceRate) {
    // This is a simplified representation - in practice, statistical power depends on many factors
    const powerElement = $('.progress-bar');
    const powerPercentage = Math.min(95, 60 + (significanceRate * 0.5)); // Simplified calculation

    powerElement.css('width', powerPercentage + '%');

    let powerText = 'Sample Size: ';
    if (powerPercentage > 80) powerText += 'Excellent';
    else if (powerPercentage > 70) powerText += 'Good';
    else if (powerPercentage > 60) powerText += 'Adequate';
    else powerText += 'Limited';

    powerElement.text(powerText);
}

function exportResults() {
    if (Object.keys(testResults).length === 0) {
        showAlert('No test results to export. Run tests first.', 'warning');
        return;
    }

    showLoading('Preparing export...');

    // Prepare export data
    const exportData = {
        timestamp: new Date().toISOString(),
        summary: {
            total_tests: Object.keys(testResults).length,
            significant_results: significantCount,
            significance_rate: ((significantCount / Object.keys(testResults).length) * 100).toFixed(1) + '%'
        },
        tests: testResults,
        methodology: {
            significance_level: 0.05,
            confidence_level: 0.95,
            multiple_comparisons_correction: 'None applied',
            software: 'WorkWell Analytics Platform'
        },
        interpretation_guide: {
            p_value: 'Probability of observing results if null hypothesis is true',
            effect_size: 'Magnitude of difference between groups',
            confidence_interval: '95% CI for the difference'
        }
    };

    fetch('/api/export/hypothesis_summary', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(exportData)
    })
    .then(response => {
        if (!response.ok) {
            // If API endpoint doesn't exist, export client-side data
            return Promise.resolve(exportData);
        }
        return response.json();
    })
    .then(data => {
        const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `hypothesis_results_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        showAlert('Statistical results exported successfully', 'success');
    })
    .catch(error => {
        console.error('Export error:', error);
        showAlert(`Failed to export results: ${error.message}`, 'danger');
    })
    .finally(() => {
        hideLoading();
    });
}

function resetTests() {
    if (testRunning) {
        showAlert('Cannot reset while tests are running', 'warning');
        return;
    }

    testResults = {};
    significantCount = 0;

    // Reset all UI elements
    $('.badge[id$="-result"]').removeClass('bg-success bg-danger').addClass('bg-secondary').text('Not Run');
    $('[id$="-interpretation"]').text('Click "Run Test" to see results');
    $('[id$="-chart"]').empty();

    // Reset all test statistics
    $('[id$="-samples"], [id$="-statistic"], [id$="-pvalue"], [id$="-effect"], [id$="-male"], [id$="-female"], [id$="-chi2"], [id$="-cramers"], [id$="-fstat"], [id$="-eta"], [id$="-count"], [id$="-correlation"], [id$="-direction"], [id$="-strength"]').text('-');

    $('#significant-count').text('0');
    $('#key-findings-list').html('<li>Run tests to see findings...</li>');

    // Reset progress bar
    $('.progress-bar').css('width', '85%').text('Sample Size: Adequate');

    showAlert('All tests have been reset', 'info');
}

function loadInitialResults() {
    // Check for pre-computed results from the server
    fetch('/api/hypothesis/summary')
        .then(response => {
            if (response.ok) {
                return response.json();
            }
            throw new Error('No pre-computed results');
        })
        .then(data => {
            if (data && data.tests) {
                console.log('Loading pre-computed hypothesis test results');

                // Load each test result
                for (let testType in data.tests) {
                    testResults[testType] = data.tests[testType];
                    updateTestUI(testType, data.tests[testType]);
                }

                updateSummary();
                showAlert('Pre-computed test results loaded', 'info');
            }
        })
        .catch(error => {
            console.log('No pre-computed results available, starting fresh');
        });
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

    const hideDelay = type === 'danger' ? 10000 : 5000;
    setTimeout(() => {
        alertContainer.find('.alert').last().alert('close');
    }, hideDelay);

    const alerts = alertContainer.find('.alert');
    if (alerts.length > 3) {
        alerts.first().alert('close');
    }
}

function showLoading(message = 'Processing...') {
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

function initializeUIEnhancements() {
    // Initialize tooltips
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // Add loading states to buttons
    $('.btn').on('click', function() {
        const btn = $(this);
        if (!btn.hasClass('no-loading') && !btn.prop('disabled')) {
            const originalText = btn.html();
            btn.data('original-text', originalText);
        }
    });

    // Add statistical interpretation help
    addStatisticalHelp();
}

function addStatisticalHelp() {
    // Add help tooltips to statistical terms
    const helpTooltips = {
        'P-value': 'The probability of obtaining test results at least as extreme as the observed results, assuming the null hypothesis is true.',
        'Effect Size': 'A quantitative measure of the magnitude of the experimental effect.',
        'Confidence Interval': 'A range of values that is likely to contain the true population parameter.',
        'Statistical Power': 'The probability of correctly rejecting a false null hypothesis.'
    };

    // This would add tooltips to statistical terms in the UI
    // Implementation depends on your specific HTML structure
}

function setupKeyboardShortcuts() {
    $(document).on('keydown', function(e) {
        // Ctrl/Cmd + Enter to run all tests
        if ((e.ctrlKey || e.metaKey) && e.keyCode === 13) {
            e.preventDefault();
            runAllTests();
        }

        // Ctrl/Cmd + R to reset (prevent browser refresh)
        if ((e.ctrlKey || e.metaKey) && e.keyCode === 82) {
            e.preventDefault();
            resetTests();
        }

        // Ctrl/Cmd + E to export
        if ((e.ctrlKey || e.metaKey) && e.keyCode === 69) {
            e.preventDefault();
            exportResults();
        }

        // Escape to cancel/hide loading
        if (e.keyCode === 27) {
            hideLoading();
        }

        // Number keys 1-5 to run individual tests
        if (e.altKey && e.keyCode >= 49 && e.keyCode <= 53) {
            e.preventDefault();
            const testTypes = ['remote_burnout', 'gender_balance', 'age_mental_health', 'industry_hours', 'salary_isolation'];
            const testIndex = e.keyCode - 49;
            if (testIndex < testTypes.length) {
                runTest(testTypes[testIndex]);
            }
        }
    });
}

// Export functions for external use
window.HypothesisModule = {
    runTest,
    runAllTests,
    exportResults,
    resetTests,
    showAlert,
    showLoading,
    hideLoading,
    getTestResults: () => testResults,
    getSignificantCount: () => significantCount
};

// Add keyboard shortcut info to console
console.log('Keyboard shortcuts: Ctrl+Enter (Run all), Ctrl+R (Reset), Ctrl+E (Export), Alt+1-5 (Individual tests), Esc (Cancel)');