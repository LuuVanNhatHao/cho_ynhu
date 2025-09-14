// Application State Management
const AppState = {
    dataLoaded: false,
    currentSection: 'dashboard',
    analysisResults: {},
    chartInstances: {},
    recommendations: []
};

// Utility Functions
const Utils = {
    showLoading: (show = true) => {
        document.getElementById('loadingContainer').style.display = show ? 'flex' : 'none';
    },

    showAutoLoadingStatus: (status, message) => {
        const statusElement = document.getElementById('autoLoadingStatus');
        const statusIndicator = statusElement.querySelector('.status-indicator');
        const statusText = document.getElementById('statusText');

        statusElement.style.display = 'flex';
        statusText.textContent = message;

        // Update indicator class
        statusIndicator.className = 'status-indicator ' + status;

        // Auto-hide after 5 seconds for success/error
        if (status === 'success' || status === 'error') {
            setTimeout(() => {
                statusElement.style.display = 'none';
            }, 5000);
        }
    },

    showAlert: (message, type = 'info', duration = 5000) => {
        const alertContainer = document.getElementById('alertContainer');
        const alertId = 'alert_' + Date.now();

        const alertHtml = `
            <div id="${alertId}" class="alert-glass alert-${type} animate__animated animate__slideInDown">
                <div class="d-flex align-items-center">
                    <i class="fas fa-${Utils.getAlertIcon(type)} me-3"></i>
                    <div>${message}</div>
                    <button class="btn btn-sm btn-glass ms-auto" onclick="this.parentElement.parentElement.remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
        `;

        alertContainer.insertAdjacentHTML('beforeend', alertHtml);

        if (duration > 0) {
            setTimeout(() => {
                const alertElement = document.getElementById(alertId);
                if (alertElement) {
                    alertElement.classList.add('animate__fadeOutUp');
                    setTimeout(() => alertElement.remove(), 500);
                }
            }, duration);
        }
    },

    getAlertIcon: (type) => {
        const icons = {
            'success': 'check-circle',
            'error': 'exclamation-triangle',
            'warning': 'exclamation-circle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    },

    formatNumber: (num) => {
        return new Intl.NumberFormat('vi-VN').format(num);
    },

    animateValue: (element, start, end, duration = 1000) => {
        const range = end - start;
        let startTime;

        const step = (timestamp) => {
            if (!startTime) startTime = timestamp;
            const progress = Math.min((timestamp - startTime) / duration, 1);
            const value = Math.floor(progress * range + start);
            element.textContent = Utils.formatNumber(value);

            if (progress < 1) {
                requestAnimationFrame(step);
            }
        };

        requestAnimationFrame(step);
    }
};

// Navigation Management
const Navigation = {
    init: () => {
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
                Navigation.switchSection(section);
            });
        });
    },

    switchSection: (section) => {
        // Update active nav item
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-section="${section}"]`).classList.add('active');

        // Hide all sections
        document.querySelectorAll('.content-section').forEach(sec => {
            sec.style.display = 'none';
        });

        // Show selected section
        document.getElementById(`${section}-section`).style.display = 'block';

        AppState.currentSection = section;

        // Load section-specific content if needed
        Navigation.loadSectionContent(section);
    },

    loadSectionContent: (section) => {
        switch(section) {
            case 'visualizations':
                if (AppState.analysisResults.visualizations) {
                    Visualizations.displayAll();
                }
                break;
            case 'ml-analysis':
                if (AppState.analysisResults.mlAnalysis) {
                    MLAnalysis.display();
                }
                break;
            case 'clustering':
                if (AppState.analysisResults.clustering) {
                    Clustering.display();
                }
                break;
            case 'statistical-tests':
                if (AppState.analysisResults.statisticalTests) {
                    StatisticalTests.display();
                }
                break;
            case 'recommendations':
                if (AppState.recommendations.length > 0) {
                    Recommendations.display();
                }
                break;
        }
    }
};

// Data Management
const DataManager = {
    autoLoadData: async () => {
        Utils.showAutoLoadingStatus('loading', 'Đang tự động tải dữ liệu...');

        try {
            const response = await fetch('/api/initial_load');
            const data = await response.json();

            if (data.status === 'success') {
                AppState.dataLoaded = true;

                // Update quick stats
                DataManager.updateQuickStats(data.basic_stats);

                // Display data overview
                DataManager.displayDataOverview(data);

                // Display initial plots if available
                if (data.initial_plots) {
                    DataManager.displayInitialPlots(data.initial_plots);
                }

                Utils.showAutoLoadingStatus('success', '✅ Dữ liệu đã được tải tự động!');
                Utils.showAlert('🎉 Dữ liệu đã sẵn sàng! Bạn có thể bắt đầu phân tích ngay.', 'success');

                return true;
            } else {
                Utils.showAutoLoadingStatus('error', '❌ Không thể tải dữ liệu tự động');
                Utils.showAlert(`⚠️ ${data.message}`, 'warning');
                return false;
            }
        } catch (error) {
            Utils.showAutoLoadingStatus('error', '❌ Lỗi kết nối server');
            Utils.showAlert(`❌ Lỗi kết nối: ${error.message}`, 'error');
            return false;
        }
    },

    reloadData: async () => {
        Utils.showLoading(true);

        try {
            const response = await fetch('/api/initial_load');
            const data = await response.json();

            if (data.status === 'success') {
                AppState.dataLoaded = true;
                Utils.showAlert(`✅ ${data.message}`, 'success');

                // Update quick stats
                DataManager.updateQuickStats(data.basic_stats);

                // Show data overview
                DataManager.displayDataOverview(data);

                // Display initial plots
                if (data.initial_plots) {
                    DataManager.displayInitialPlots(data.initial_plots);
                }

                return true;
            } else {
                Utils.showAlert(`❌ ${data.message}`, 'error');
                return false;
            }
        } catch (error) {
            Utils.showAlert(`❌ Lỗi kết nối: ${error.message}`, 'error');
            return false;
        } finally {
            Utils.showLoading(false);
        }
    },

    updateQuickStats: (stats) => {
        // Animate stats values
        const totalRecords = document.getElementById('totalRecords');
        const totalColumns = document.getElementById('totalColumns');
        const missingValues = document.getElementById('missingValues');
        const memoryUsage = document.getElementById('memoryUsage');

        if (totalRecords) Utils.animateValue(totalRecords, 0, stats.total_records);
        if (totalColumns) Utils.animateValue(totalColumns, 0, stats.total_columns);
        if (missingValues) Utils.animateValue(missingValues, 0, stats.missing_values);
        if (memoryUsage) memoryUsage.textContent = stats.memory_usage;
    },

    displayDataOverview: (data) => {
        const container = document.getElementById('dataOverviewContent');
        if (!container) return;

        const overviewHtml = `
            <div class="row">
                <div class="col-md-6">
                    <h5 style="color: white;">Cấu trúc dữ liệu</h5>
                    <div class="table-responsive">
                        <table class="table table-dark table-striped">
                            <thead>
                                <tr>
                                    <th>Cột</th>
                                    <th>Kiểu dữ liệu</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${data.columns.map(col => `
                                    <tr>
                                        <td>${col}</td>
                                        <td><span class="badge bg-secondary">${DataManager.getColumnType(col)}</span></td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
                <div class="col-md-6">
                    <h5 style="color: white;">Mẫu dữ liệu</h5>
                    <div class="table-responsive">
                        <table class="table table-dark table-striped">
                            <thead>
                                <tr>
                                    ${data.columns.slice(0, 5).map(col => `<th>${col}</th>`).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                ${data.sample_data.slice(0, 5).map(row => `
                                    <tr>
                                        ${data.columns.slice(0, 5).map(col => `
                                            <td>${DataManager.truncateText(row[col], 20)}</td>
                                        `).join('')}
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;

        container.innerHTML = overviewHtml;
    },

    displayInitialPlots: (plots) => {
        if (plots.dashboard_overview) {
            const container = document.getElementById('dashboardChart');
            const plotArea = document.getElementById('dashboardPlotArea');

            if (container && plotArea) {
                container.style.display = 'block';
                const plotData = JSON.parse(plots.dashboard_overview);
                Plotly.newPlot('dashboardPlotArea', plotData.data, plotData.layout, {responsive: true});
            }
        }
    },

    getColumnType: (column) => {
        const numericColumns = ['Age', 'Hours_Per_Week', 'Work_Life_Balance_Score', 'Social_Isolation_Score'];
        return numericColumns.includes(column) ? 'Số' : 'Văn bản';
    },

    truncateText: (text, maxLength) => {
        if (!text) return 'N/A';
        return text.toString().length > maxLength ?
            text.toString().substring(0, maxLength) + '...' :
            text.toString();
    }
};

// Data Analysis Module
const DataAnalysis = {
    startFullAnalysis: async () => {
        if (!AppState.dataLoaded) {
            Utils.showAlert('Vui lòng đợi dữ liệu được tải', 'warning');
            return;
        }

        Utils.showLoading(true);
        Utils.showAlert('🚀 Bắt đầu phân tích toàn diện...', 'info');

        try {
            // Load all analyses in parallel
            const promises = [
                DataAnalysis.loadVisualizations(),
                DataAnalysis.loadMLAnalysis(),
                DataAnalysis.loadClustering(),
                DataAnalysis.loadStatisticalTests()
            ];

            await Promise.all(promises);

            Utils.showAlert('✅ Phân tích hoàn tất! Tất cả kết quả đã sẵn sàng.', 'success');

            // Auto-switch to ML analysis section
            Navigation.switchSection('ml-analysis');

        } catch (error) {
            Utils.showAlert(`❌ Lỗi phân tích: ${error.message}`, 'error');
        } finally {
            Utils.showLoading(false);
        }
    },

    loadVisualizations: async () => {
        try {
            const response = await fetch('/api/advanced_visualizations');
            const data = await response.json();

            if (data.status === 'success') {
                AppState.analysisResults.visualizations = data.plots;
                return true;
            }
            return false;
        } catch (error) {
            console.error('Error loading visualizations:', error);
            return false;
        }
    },

    loadMLAnalysis: async () => {
        try {
            const response = await fetch('/api/advanced_analysis');
            const data = await response.json();

            if (data.status === 'success') {
                AppState.analysisResults.mlAnalysis = data;
                AppState.recommendations = data.recommendations || [];
                return true;
            }
            return false;
        } catch (error) {
            console.error('Error loading ML analysis:', error);
            return false;
        }
    },

    loadClustering: async () => {
        try {
            const response = await fetch('/api/clustering_analysis');
            const data = await response.json();

            if (data.status === 'success') {
                AppState.analysisResults.clustering = data;
                return true;
            }
            return false;
        } catch (error) {
            console.error('Error loading clustering:', error);
            return false;
        }
    },

    loadStatisticalTests: async () => {
        try {
            const response = await fetch('/api/statistical_tests');
            const data = await response.json();

            if (data.status === 'success') {
                AppState.analysisResults.statisticalTests = data.tests;
                return true;
            }
            return false;
        } catch (error) {
            console.error('Error loading statistical tests:', error);
            return false;
        }
    }
};

// Visualizations Module
const Visualizations = {
    displayAll: () => {
        const plots = AppState.analysisResults.visualizations;
        if (!plots) return;

        // Display 3D Scatter
        if (plots.scatter_3d) {
            const plotData = JSON.parse(plots.scatter_3d);
            Plotly.newPlot('viz-3d-scatter', plotData.data, plotData.layout, {responsive: true});
        }

        // Display Heatmap
        if (plots.correlation_heatmap) {
            const plotData = JSON.parse(plots.correlation_heatmap);
            Plotly.newPlot('viz-heatmap', plotData.data, plotData.layout, {responsive: true});
        }

        // Display Sunburst
        if (plots.sunburst) {
            const plotData = JSON.parse(plots.sunburst);
            Plotly.newPlot('viz-sunburst', plotData.data, plotData.layout, {responsive: true});
        }

        // Display Violin
        if (plots.violin_arrangement) {
            const plotData = JSON.parse(plots.violin_arrangement);
            Plotly.newPlot('viz-violin', plotData.data, plotData.layout, {responsive: true});
        }
    }
};

const InsightsModule = {
    // Replace ML Analysis with Insights Dashboard
    displayInsightsDashboard: async () => {
        try {
            const response = await fetch('/api/enhanced_analytics');
            const data = await response.json();

            if (data.status === 'success') {
                InsightsModule.renderRiskDashboard(data.risk_analysis);
                InsightsModule.renderKeyInsights(data);
                InsightsModule.renderActionableRecommendations(data);
                InsightsModule.renderVisualizations(data.visualizations);
            }
        } catch (error) {
            console.error('Error loading insights:', error);
        }
    },

    renderRiskDashboard: (riskData) => {
        const container = document.getElementById('ml-analysis-section');

        const html = `
            <div class="insights-dashboard">
                <h3 class="chart-title">📊 Risk Assessment & Key Insights</h3>
                
                <!-- Overall Risk Score -->
                <div class="risk-score-container">
                    <div class="composite-risk-card ${InsightsModule.getRiskClass(riskData.composite_risk)}">
                        <h4>Mức độ rủi ro tổng thể</h4>
                        <div class="risk-meter">
                            <div class="risk-value">${riskData.composite_risk.toFixed(1)}%</div>
                            <div class="risk-level">${riskData.risk_level}</div>
                        </div>
                        <div class="risk-bar">
                            <div class="risk-fill" style="width: ${riskData.composite_risk}%"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Individual Risk Factors -->
                <div class="row mt-4">
                    ${Object.entries(riskData.risk_factors).map(([factor, value]) => `
                        <div class="col-md-3">
                            <div class="risk-factor-card">
                                <div class="risk-icon">
                                    <i class="fas fa-${InsightsModule.getRiskIcon(factor)}"></i>
                                </div>
                                <h5>${InsightsModule.formatRiskName(factor)}</h5>
                                <div class="risk-percentage ${InsightsModule.getRiskClass(value)}">
                                    ${value.toFixed(1)}%
                                </div>
                                <div class="risk-indicator">
                                    <div class="indicator-bar">
                                        <div class="indicator-fill" style="width: ${value}%"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
                
                <!-- Visualization Container -->
                <div id="risk-gauge-chart" class="mt-4"></div>
            </div>
        `;

        container.innerHTML = html;
    },

    renderKeyInsights: (data) => {
        const insights = [];

        // Extract key insights from data
        if (data.demographic_insights) {
            if (data.demographic_insights.high_risk_industries) {
                const topIndustry = Object.entries(data.demographic_insights.high_risk_industries)[0];
                insights.push({
                    type: 'warning',
                    icon: 'industry',
                    title: 'Ngành có rủi ro cao nhất',
                    content: `${topIndustry[0]} với ${topIndustry[1].toFixed(1)}% nhân viên có vấn đề sức khỏe tinh thần`
                });
            }
        }

        if (data.anomaly_detection) {
            insights.push({
                type: 'info',
                icon: 'exclamation-triangle',
                title: 'Phát hiện bất thường',
                content: `${data.anomaly_detection.percentage.toFixed(1)}% nhân viên có chỉ số bất thường cần theo dõi đặc biệt`
            });
        }

        if (data.predictive_indicators && data.predictive_indicators.top_predictors) {
            insights.push({
                type: 'success',
                icon: 'chart-line',
                title: 'Yếu tố dự báo quan trọng nhất',
                content: data.predictive_indicators.top_predictors[0].replace('_', ' ')
            });
        }

        // Render insights cards
        const container = document.createElement('div');
        container.className = 'insights-cards-container mt-4';
        container.innerHTML = `
            <h4 class="text-white mb-3">💡 Key Insights</h4>
            <div class="row">
                ${insights.map(insight => `
                    <div class="col-md-4">
                        <div class="insight-card ${insight.type}">
                            <div class="insight-icon">
                                <i class="fas fa-${insight.icon}"></i>
                            </div>
                            <h5>${insight.title}</h5>
                            <p>${insight.content}</p>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;

        document.getElementById('ml-analysis-section').appendChild(container);
    },

    renderActionableRecommendations: (data) => {
        const recommendations = [];

        // Generate recommendations based on insights
        if (data.risk_analysis.risk_factors.work_overload > 50) {
            recommendations.push({
                priority: 'high',
                title: 'Giảm tải công việc',
                action: 'Triển khai chính sách giới hạn giờ làm thêm và phân bổ lại khối lượng công việc',
                impact: 'Giảm 30% nguy cơ burnout',
                timeline: '1-2 tháng'
            });
        }

        if (data.risk_analysis.risk_factors.social_isolation > 40) {
            recommendations.push({
                priority: 'medium',
                title: 'Tăng cường kết nối',
                action: 'Tổ chức team building hàng tháng và tạo không gian làm việc chung',
                impact: 'Cải thiện 25% chỉ số hạnh phúc',
                timeline: '2-3 tuần'
            });
        }

        const container = document.createElement('div');
        container.className = 'recommendations-container mt-4';
        container.innerHTML = `
            <h4 class="text-white mb-3">🎯 Khuyến nghị hành động</h4>
            <div class="action-cards">
                ${recommendations.map((rec, index) => `
                    <div class="action-card priority-${rec.priority}">
                        <div class="action-number">${index + 1}</div>
                        <div class="action-content">
                            <h5>${rec.title}</h5>
                            <p class="action-description">${rec.action}</p>
                            <div class="action-metrics">
                                <span class="metric">
                                    <i class="fas fa-chart-line"></i> ${rec.impact}
                                </span>
                                <span class="metric">
                                    <i class="fas fa-clock"></i> ${rec.timeline}
                                </span>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;

        document.getElementById('ml-analysis-section').appendChild(container);
    },

    renderVisualizations: (plots) => {
        // Render enhanced visualizations
        if (plots.risk_dashboard) {
            const plotData = JSON.parse(plots.risk_dashboard);
            Plotly.newPlot('risk-gauge-chart', plotData.data, plotData.layout, {responsive: true});
        }

        // Add more visualization containers as needed
        const vizContainer = document.createElement('div');
        vizContainer.className = 'enhanced-viz-container mt-4';
        vizContainer.id = 'enhanced-visualizations';

        document.getElementById('ml-analysis-section').appendChild(vizContainer);

        // Render additional plots
        Object.entries(plots).forEach(([key, plotJson]) => {
            if (key !== 'risk_dashboard') {
                const plotDiv = document.createElement('div');
                plotDiv.id = `plot-${key}`;
                plotDiv.className = 'plot-container mb-4';
                vizContainer.appendChild(plotDiv);

                const plotData = JSON.parse(plotJson);
                Plotly.newPlot(plotDiv.id, plotData.data, plotData.layout, {responsive: true});
            }
        });
    },

    // Helper functions
    getRiskClass: (value) => {
        if (value > 60) return 'risk-high';
        if (value > 40) return 'risk-medium';
        return 'risk-low';
    },

    getRiskIcon: (factor) => {
        const icons = {
            'work_overload': 'briefcase',
            'social_isolation': 'user-friends',
            'work_life_imbalance': 'balance-scale',
            'burnout_risk': 'fire'
        };
        return icons[factor] || 'exclamation-triangle';
    },

    formatRiskName: (factor) => {
        const names = {
            'work_overload': 'Quá tải công việc',
            'social_isolation': 'Cô lập xã hội',
            'work_life_imbalance': 'Mất cân bằng',
            'burnout_risk': 'Nguy cơ kiệt sức'
        };
        return names[factor] || factor.replace('_', ' ');
    }
};

// Clustering Module
const Clustering = {
    display: () => {
        const data = AppState.analysisResults.clustering;
        if (!data) return;

        // Display clustering results
        const resultsContainer = document.getElementById('clustering-results');
        if (resultsContainer) {
            resultsContainer.innerHTML = Clustering.renderResults(data.clustering_results);
        }

        // Display visualization
        const vizContainer = document.getElementById('clustering-visualization');
        if (vizContainer && data.visualization) {
            const plotData = JSON.parse(data.visualization);
            Plotly.newPlot('clustering-visualization', plotData.data, plotData.layout, {responsive: true});
        }
    },

    renderResults: (results) => {
        return `
            <h4 style="color: white;">Kết quả phân nhóm</h4>
            <p style="color: rgba(255,255,255,0.9);">
                Số nhóm tối ưu: <strong>${results.optimal_k}</strong>
            </p>
            <div class="row">
                ${Object.entries(results.cluster_summary).map(([cluster, summary]) => `
                    <div class="col-md-4">
                        <div class="glass-container p-3">
                            <h5 style="color: white;">${cluster.replace('_', ' ')}</h5>
                            <p style="color: rgba(255,255,255,0.8);">
                                Kích thước: ${summary.size} người<br>
                                Tuổi TB: ${summary.avg_age.toFixed(1)}<br>
                                Giờ làm/tuần: ${summary.avg_hours.toFixed(1)}<br>
                                WLB Score: ${summary.avg_work_life_balance.toFixed(2)}<br>
                                Isolation Score: ${summary.avg_isolation.toFixed(2)}
                            </p>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }
};

// Statistical Tests Module
const StatisticalTests = {
    display: () => {
        const tests = AppState.analysisResults.statisticalTests;
        if (!tests) return;

        const container = document.getElementById('statistical-tests-results');
        if (container) {
            container.innerHTML = StatisticalTests.renderTests(tests);
        }
    },

    renderTests: (tests) => {
        return `
            <h4 style="color: white;">Kết quả kiểm định thống kê</h4>
            <div class="row">
                ${Object.entries(tests).map(([key, test]) => `
                    <div class="col-md-12 mb-3">
                        <div class="glass-container p-3">
                            <h5 style="color: white;">${test.test_name}</h5>
                            <div style="color: rgba(255,255,255,0.9);">
                                <p>Thống kê: ${test.statistic.toFixed(4)}</p>
                                <p>P-value: ${test.p_value.toFixed(4)}</p>
                                <p class="mb-0">
                                    <strong>Kết luận:</strong> 
                                    <span class="badge ${test.p_value < 0.05 ? 'bg-success' : 'bg-warning'}">
                                        ${test.interpretation}
                                    </span>
                                </p>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }
};

// Recommendations Module
const Recommendations = {
    display: () => {
        const recommendations = AppState.recommendations;
        if (!recommendations || recommendations.length === 0) {
            // Try to get from ML analysis
            if (AppState.analysisResults.mlAnalysis && AppState.analysisResults.mlAnalysis.recommendations) {
                recommendations.push(...AppState.analysisResults.mlAnalysis.recommendations);
            }
        }

        const container = document.getElementById('recommendations-content');
        if (container) {
            container.innerHTML = Recommendations.render(recommendations);
        }
    },

    render: (recommendations) => {
        if (!recommendations || recommendations.length === 0) {
            return '<p style="color: white;">Chưa có khuyến nghị. Vui lòng chạy phân tích ML trước.</p>';
        }

        return `
            <div class="row">
                ${recommendations.map((rec, index) => `
                    <div class="col-md-12 mb-3">
                        <div class="recommendation-card priority-${(rec.priority || 'medium').toLowerCase()}">
                            <div class="d-flex align-items-start">
                                <div class="me-3">
                                    <span class="badge bg-${Recommendations.getPriorityColor(rec.priority)}">
                                        ${rec.priority}
                                    </span>
                                </div>
                                <div class="flex-grow-1">
                                    <h5 style="color: white;">
                                        <i class="fas fa-${Recommendations.getAreaIcon(rec.area)}"></i> 
                                        ${rec.area}
                                    </h5>
                                    <p style="color: rgba(255,255,255,0.9);">${rec.recommendation}</p>
                                    <div class="mt-2">
                                        <small style="color: rgba(255,255,255,0.7);">
                                            <i class="fas fa-chart-line"></i> 
                                            Tác động dự kiến: <strong>${rec.expected_impact}</strong>
                                        </small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    },

    getPriorityColor: (priority) => {
        const colors = {
            'Cao': 'danger',
            'Trung bình': 'warning',
            'Thấp': 'success'
        };
        return colors[priority] || 'secondary';
    },

    getAreaIcon: (area) => {
        const icons = {
            'Cân bằng công việc-cuộc sống': 'balance-scale',
            'Tương tác xã hội': 'users',
            'Quản lý thời gian': 'clock'
        };
        return icons[area] || 'lightbulb';
    }
};

// Report Export Module
const ReportExporter = {
    exportReport: () => {
        const reportData = ReportExporter.generateReportData();
        const reportHtml = ReportExporter.generateReportHTML(reportData);
        ReportExporter.downloadReport(reportHtml);
    },

    generateReportData: () => {
        return {
            timestamp: new Date().toLocaleString('vi-VN'),
            dataLoaded: AppState.dataLoaded,
            analysisResults: AppState.analysisResults,
            recommendations: AppState.recommendations
        };
    },

    generateReportHTML: (data) => {
        return `
            <!DOCTYPE html>
            <html lang="vi">
            <head>
                <meta charset="UTF-8">
                <title>Báo cáo Phân tích Sức khỏe Tinh thần</title>
                <style>
                    body { 
                        font-family: 'Segoe UI', sans-serif; 
                        margin: 40px;
                        line-height: 1.6;
                    }
                    .header { 
                        text-align: center; 
                        margin-bottom: 40px;
                        padding: 20px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        border-radius: 10px;
                    }
                    .section { 
                        margin: 30px 0;
                        padding: 20px;
                        border-left: 4px solid #667eea;
                        background: #f8f9fa;
                    }
                    h2 { color: #333; }
                    .metric { 
                        display: inline-block;
                        margin: 10px;
                        padding: 10px;
                        background: white;
                        border-radius: 5px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Báo cáo Phân tích Sức khỏe Tinh thần Người lao động</h1>
                    <p>Được tạo vào: ${data.timestamp}</p>
                </div>
                
                <div class="section">
                    <h2>Tóm tắt Executive</h2>
                    <p>Báo cáo này cung cấp phân tích toàn diện về các yếu tố ảnh hưởng đến sức khỏe tinh thần của người lao động, 
                    sử dụng các thuật toán Machine Learning tiên tiến để đưa ra những insights có giá trị.</p>
                </div>
                
                <div class="section">
                    <h2>Kết quả Machine Learning</h2>
                    ${data.analysisResults.mlAnalysis ? `
                        <p>Mô hình tốt nhất: <strong>${data.analysisResults.mlAnalysis.best_model}</strong></p>
                        <p>Độ chính xác: <strong>${(data.analysisResults.mlAnalysis.models_performance[data.analysisResults.mlAnalysis.best_model].accuracy * 100).toFixed(2)}%</strong></p>
                    ` : '<p>Chưa có kết quả phân tích ML</p>'}
                </div>
                
                <div class="section">
                    <h2>Khuyến nghị chiến lược</h2>
                    ${data.recommendations && data.recommendations.length > 0 ? 
                        data.recommendations.map(rec => `
                            <div style="margin-bottom: 15px;">
                                <h3>${rec.area}</h3>
                                <p>${rec.recommendation}</p>
                                <p><em>Tác động dự kiến: ${rec.expected_impact}</em></p>
                            </div>
                        `).join('') : 
                        '<p>Chưa có khuyến nghị</p>'
                    }
                </div>
            </body>
            </html>
        `;
    },

    downloadReport: (htmlContent) => {
        const blob = new Blob([htmlContent], { type: 'text/html' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `mental_health_report_${new Date().getTime()}.html`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);

        Utils.showAlert('✅ Báo cáo đã được tải xuống thành công!', 'success');
    }
};

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    // Initialize navigation
    Navigation.init();

    // Auto-load data on page load
    await DataManager.autoLoadData();

    // Event listeners
    document.getElementById('reloadDataBtn').addEventListener('click', DataManager.reloadData);
    document.getElementById('startAnalysisBtn').addEventListener('click', DataAnalysis.startFullAnalysis);
    document.getElementById('exportReportBtn').addEventListener('click', ReportExporter.exportReport);

    // Feature cards click events
    document.querySelectorAll('.feature-card').forEach(card => {
        card.addEventListener('click', (e) => {
            const feature = e.currentTarget.dataset.feature;
            switch(feature) {
                case 'advanced-viz':
                    Navigation.switchSection('visualizations');
                    break;
                case 'ml-models':
                    Navigation.switchSection('ml-analysis');
                    break;
                case 'clustering':
                    Navigation.switchSection('clustering');
                    break;
            }
        });
    });
});


