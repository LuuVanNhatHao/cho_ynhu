// Statistical Tests Module
// Handles statistical analysis and hypothesis testing

const StatisticalTests = {
    // Setup control panel
    setupControls: () => {
        if (!AppState.schemaInfo) return;

        const var1Select = document.getElementById('var1Select');
        const var2Select = document.getElementById('var2Select');

        if (!var1Select || !var2Select) return;

        // Clear existing options
        var1Select.innerHTML = '<option value="">Select Variable 1</option>';
        var2Select.innerHTML = '<option value="">Select Variable 2</option>';

        // Populate variable options
        Object.keys(AppState.schemaInfo.column_info).forEach(col => {
            const option1 = document.createElement('option');
            option1.value = col;
            option1.textContent = col.replace(/_/g, ' ');
            var1Select.appendChild(option1);

            const option2 = document.createElement('option');
            option2.value = col;
            option2.textContent = col.replace(/_/g, ' ');
            var2Select.appendChild(option2);
        });

        // Add event listeners for variable selection
        var1Select.addEventListener('change', StatisticalTests.updateTestInfo);
        var2Select.addEventListener('change', StatisticalTests.updateTestInfo);

        // Add run button listener
        const runBtn = document.getElementById('runTestsBtn');
        if (runBtn) {
            runBtn.addEventListener('click', () => {
                StatisticalTests.runCustomTests();
            });
        }

        // Load default tests
        StatisticalTests.loadDefaultTests();
    },

    // Update test information based on selected variables
    updateTestInfo: () => {
        const var1 = document.getElementById('var1Select')?.value;
        const var2 = document.getElementById('var2Select')?.value;
        const infoContainer = document.getElementById('testTypeInfo');
        const descriptionContainer = document.getElementById('testTypeDescription');

        if (!var1 || !var2 || !AppState.schemaInfo || !infoContainer || !descriptionContainer) return;

        const var1Info = AppState.schemaInfo.column_info[var1];
        const var2Info = AppState.schemaInfo.column_info[var2];

        const var1IsNumeric = var1Info.dtype === 'int64' || var1Info.dtype === 'float64';
        const var2IsNumeric = var2Info.dtype === 'int64' || var2Info.dtype === 'float64';

        let testType = '';
        let description = '';

        if (!var1IsNumeric && !var2IsNumeric) {
            testType = 'Chi-square Test';
            description = 'Tests for association between two categorical variables. Null hypothesis: No association exists.';
        } else if (var1IsNumeric && !var2IsNumeric) {
            const uniqueCount = var2Info.unique_count;
            if (uniqueCount === 2) {
                testType = 'Independent T-test';
                description = 'Compares means between two groups. Null hypothesis: No difference in means.';
            } else if (uniqueCount > 2) {
                testType = 'One-way ANOVA';
                description = 'Compares means across multiple groups. Null hypothesis: All group means are equal.';
            }
        } else if (!var1IsNumeric && var2IsNumeric) {
            const uniqueCount = var1Info.unique_count;
            if (uniqueCount === 2) {
                testType = 'Independent T-test';
                description = 'Compares means between two groups. Null hypothesis: No difference in means.';
            } else if (uniqueCount > 2) {
                testType = 'One-way ANOVA';
                description = 'Compares means across multiple groups. Null hypothesis: All group means are equal.';
            }
        } else if (var1IsNumeric && var2IsNumeric) {
            testType = 'Pearson Correlation';
            description = 'Tests for linear relationship between two continuous variables. Null hypothesis: No correlation exists.';
        }

        if (testType) {
            descriptionContainer.innerHTML = `
                <strong>Test Type:</strong> ${testType}<br>
                <strong>Description:</strong> ${description}
            `;
            infoContainer.style.display = 'block';
        } else {
            infoContainer.style.display = 'none';
        }
    },

    // Load default statistical tests
    loadDefaultTests: async () => {
        try {
            await DataAnalysis.loadStatisticalTests();
        } catch (error) {
            console.error('Error loading default tests:', error);
            const container = document.getElementById('defaultTestsContent');
            if (container) {
                container.innerHTML = '<div class="alert alert-danger">Failed to load statistical tests</div>';
            }
        }
    },

    // Run custom statistical tests
    runCustomTests: async () => {
        const var1 = document.getElementById('var1Select')?.value;
        const var2 = document.getElementById('var2Select')?.value;

        if (!var1 || !var2) {
            Utils.showAlert('Please select both variables for testing', 'warning');
            return;
        }

        if (var1 === var2) {
            Utils.showAlert('Please select different variables for testing', 'warning');
            return;
        }

        try {
            const result = await DataAnalysis.loadStatisticalTests(var1, var2);
            if (result) {
                const resultsContainer = document.getElementById('customTestResults');
                if (resultsContainer) {
                    resultsContainer.style.display = 'block';
                }
                Utils.showAlert('Custom statistical test completed successfully', 'success');
            }
        } catch (error) {
            Utils.handleApiError(error);
        }
    },

    // Render default test results
    renderDefaultTests: (tests) => {
        const container = document.getElementById('defaultTestsContent');
        if (!container) return;

        let html = '<div class="row">';

        Object.entries(tests).forEach(([testKey, testData]) => {
            if (testKey === 'error') {
                html += `<div class="col-12"><div class="alert alert-danger">${testData}</div></div>`;
                return;
            }

            const isSignificant = testData.p_value < (testData.alpha || 0.05);

            html += `
                <div class="col-md-6 mb-4">
                    <div class="stats-card">
                        <h6 style="color: var(--text-primary);">${testData.test_name}</h6>
                        <div class="row">
                            <div class="col-12">
                                <div class="mb-2">
                                    <strong>H₀:</strong> 
                                    <span style="font-size: 0.9rem;">${testData.hypothesis_h0}</span>
                                </div>
                                <div class="mb-2">
                                    <strong>H₁:</strong> 
                                    <span style="font-size: 0.9rem;">${testData.hypothesis_h1}</span>
                                </div>
                                <div class="mb-2">
                                    <strong>Test Statistic:</strong> ${testData.statistic.toFixed(4)}
                                </div>
                                <div class="mb-2">
                                    <strong>p-value:</strong> 
                                    <span class="${isSignificant ? 'text-warning' : ''}">${testData.p_value.toFixed(6)}</span>
                                </div>
                                <div class="mb-2">
                                    <strong>Result:</strong> 
                                    <span class="badge ${isSignificant ? 'bg-danger' : 'bg-success'}">
                                        ${isSignificant ? 'Reject H₀' : 'Fail to reject H₀'}
                                    </span>
                                </div>
                                <div class="mb-2">
                                    <strong>Interpretation:</strong> 
                                    <span style="font-size: 0.9rem;">${testData.interpretation}</span>
                                </div>
                                ${testData.effect_size_cohens_d ? `
                                    <div class="mb-2">
                                        <strong>Effect Size (Cohen's d):</strong> ${testData.effect_size_cohens_d.toFixed(3)}
                                        ${StatisticalTests.interpretEffectSize(testData.effect_size_cohens_d, 'cohens_d')}
                                    </div>
                                ` : ''}
                                ${testData.effect_size_eta_squared ? `
                                    <div class="mb-2">
                                        <strong>Effect Size (η²):</strong> ${testData.effect_size_eta_squared.toFixed(3)}
                                        ${StatisticalTests.interpretEffectSize(testData.effect_size_eta_squared, 'eta_squared')}
                                    </div>
                                ` : ''}
                                ${testData.effect_size ? `
                                    <div class="mb-2">
                                        <strong>Effect Size:</strong> ${testData.effect_size.toFixed(3)}
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });

        html += '</div>';
        container.innerHTML = html;
    },

    // Render custom test results
    renderCustomTests: (tests) => {
        const container = document.getElementById('customTestContent');
        if (!container) return;

        let html = '';

        Object.entries(tests).forEach(([testKey, testData]) => {
            if (testKey === 'error') {
                html += `<div class="alert alert-danger">${testData}</div>`;
                return;
            }

            const isSignificant = testData.p_value < (testData.alpha || 0.05);

            html += `
                <div class="stats-card">
                    <h5 style="color: var(--text-primary);">${testData.test_name}</h5>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="mb-2">
                                <strong>H₀:</strong> ${testData.hypothesis_h0}
                            </div>
                            <div class="mb-2">
                                <strong>H₁:</strong> ${testData.hypothesis_h1}
                            </div>
                            <div class="mb-2">
                                <strong>Test Statistic:</strong> ${testData.statistic.toFixed(4)}
                            </div>
                            <div class="mb-2">
                                <strong>p-value:</strong> 
                                <span class="${isSignificant ? 'text-warning' : ''}">${testData.p_value.toFixed(6)}</span>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="mb-2">
                                <strong>α-level:</strong> ${testData.alpha || 0.05}
                            </div>
                            <div class="mb-2">
                                <strong>Result:</strong> 
                                <span class="badge ${isSignificant ? 'bg-danger' : 'bg-success'}">
                                    ${isSignificant ? 'Reject H₀' : 'Fail to reject H₀'}
                                </span>
                            </div>
                            <div class="mb-2">
                                <strong>Interpretation:</strong> ${testData.interpretation}
                            </div>
                            ${testData.effect_size_cohens_d ? `
                                <div class="mb-2">
                                    <strong>Cohen's d:</strong> ${testData.effect_size_cohens_d.toFixed(3)}
                                    ${StatisticalTests.interpretEffectSize(testData.effect_size_cohens_d, 'cohens_d')}
                                </div>
                            ` : ''}
                            ${testData.effect_size_eta_squared ? `
                                <div class="mb-2">
                                    <strong>η²:</strong> ${testData.effect_size_eta_squared.toFixed(3)}
                                    ${StatisticalTests.interpretEffectSize(testData.effect_size_eta_squared, 'eta_squared')}
                                </div>
                            ` : ''}
                            ${testData.correlation_strength ? `
                                <div class="mb-2">
                                    <strong>Correlation Strength:</strong> ${testData.correlation_strength}
                                </div>
                            ` : ''}
                        </div>
                    </div>
                    ${testData.group1_mean !== undefined && testData.group2_mean !== undefined ? `
                        <div class="row mt-3">
                            <div class="col-12">
                                <h6 style="color: var(--text-primary);">Group Statistics:</h6>
                                <div class="mb-1">
                                    <strong>Group 1 Mean:</strong> ${testData.group1_mean.toFixed(2)} 
                                    <small class="text-muted">(n=${testData.group1_size})</small>
                                </div>
                                <div class="mb-1">
                                    <strong>Group 2 Mean:</strong> ${testData.group2_mean.toFixed(2)} 
                                    <small class="text-muted">(n=${testData.group2_size})</small>
                                </div>
                            </div>
                        </div>
                    ` : ''}
                </div>
            `;
        });

        container.innerHTML = html;
    },

    // Interpret effect size values
    interpretEffectSize: (value, type) => {
        let interpretation = '';

        if (type === 'cohens_d') {
            const absValue = Math.abs(value);
            if (absValue < 0.2) interpretation = '<small class="text-muted">(Negligible)</small>';
            else if (absValue < 0.5) interpretation = '<small class="text-info">(Small)</small>';
            else if (absValue < 0.8) interpretation = '<small class="text-warning">(Medium)</small>';
            else interpretation = '<small class="text-danger">(Large)</small>';
        } else if (type === 'eta_squared') {
            if (value < 0.01) interpretation = '<small class="text-muted">(Small)</small>';
            else if (value < 0.06) interpretation = '<small class="text-warning">(Medium)</small>';
            else interpretation = '<small class="text-danger">(Large)</small>';
        }

        return interpretation;
    },

    // Export test results
    exportTestResults: () => {
        if (!AppState.analysisResults.statisticalTests) {
            Utils.showAlert('No test results to export', 'warning');
            return;
        }

        const results = AppState.analysisResults.statisticalTests;
        const timestamp = new Date().toISOString();
        const exportData = {
            timestamp: timestamp,
            tests: results
        };

        Utils.downloadJSON(exportData, `statistical_tests_${Date.now()}.json`);
        Utils.showAlert('Test results exported successfully!', 'success');
    }
};