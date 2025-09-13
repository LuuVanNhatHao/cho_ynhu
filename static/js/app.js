/**
 * Advanced Mental Health Analytics Platform
 * Custom JavaScript for enhanced functionality
 */

// Application Configuration
const CONFIG = {
    API_BASE_URL: '/api',
    CHART_COLORS: [
        '#667eea', '#764ba2', '#f093fb', '#f5576c',
        '#4facfe', '#00f2fe', '#43e97b', '#38f9d7'
    ],
    ANIMATION_DURATION: 300,
    CHART_CONFIG: {
        responsive: true,
        displayModeBar: false,
        toImageButtonOptions: {
            format: 'png',
            filename: 'chart',
            height: 600,
            width: 800,
            scale: 1
        }
    }
};

// Application State Management
class AppStateManager {
    constructor() {
        this.state = {
            dataLoaded: false,
            currentSection: 'dashboard',
            analysisResults: {},
            chartInstances: {},
            sidebarCollapsed: false,
            theme: 'dark',
            user: {
                preferences: this.loadUserPreferences()
            }
        };

        this.listeners = {};
        this.init();
    }

    init() {
        this.createParticleBackground();
        this.initEventListeners();
        this.loadTheme();
    }

    setState(newState) {
        const oldState = { ...this.state };
        this.state = { ...this.state, ...newState };
        this.notifyListeners(oldState, this.state);
    }

    getState(key = null) {
        return key ? this.state[key] : this.state;
    }

    subscribe(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }

    notifyListeners(oldState, newState) {
        if (this.listeners['stateChange']) {
            this.listeners['stateChange'].forEach(callback => {
                callback(oldState, newState);
            });
        }
    }

    loadUserPreferences() {
        try {
            return JSON.parse(localStorage.getItem('mental_health_analytics_prefs') || '{}');
        } catch {
            return {};
        }
    }

    saveUserPreferences() {
        localStorage.setItem('mental_health_analytics_prefs', JSON.stringify(this.state.user.preferences));
    }

    loadTheme() {
        const savedTheme = this.state.user.preferences.theme || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);
        this.setState({ theme: savedTheme });
    }

    createParticleBackground() {
        const particleContainer = document.createElement('div');
        particleContainer.className = 'particle-bg';
        document.body.appendChild(particleContainer);

        for (let i = 0; i < 50; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.top = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 6 + 's';
            particle.style.animationDuration = (Math.random() * 3 + 3) + 's';
            particleContainer.appendChild(particle);
        }
    }

    initEventListeners() {
        // Handle visibility change for performance optimization
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseAnimations();
            } else {
                this.resumeAnimations();
            }
        });

        // Handle window resize
        window.addEventListener('resize', debounce(() => {
            this.handleResize();
        }, 250));
    }

    pauseAnimations() {
        document.querySelectorAll('.particle').forEach(particle => {
            particle.style.animationPlayState = 'paused';
        });
    }

    resumeAnimations() {
        document.querySelectorAll('.particle').forEach(particle => {
            particle.style.animationPlayState = 'running';
        });
    }

    handleResize() {
        // Redraw charts if they exist
        Object.values(this.state.chartInstances).forEach(chart => {
            if (chart && chart.relayout) {
                chart.relayout();
            }
        });
    }
}

// Utility Functions
const Utils = {
    showLoading(show = true) {
        const loadingContainer = document.getElementById('loadingContainer');
        if (loadingContainer) {
            loadingContainer.style.display = show ? 'flex' : 'none';
        }
    },

    showAlert(message, type = 'info', duration = 5000) {
        const alertContainer = document.getElementById('alertContainer');
        if (!alertContainer) return;

        const alertId = 'alert_' + Date.now();
        const alertElement = this.createAlertElement(alertId, message, type);

        alertContainer.appendChild(alertElement);

        // Trigger animation
        setTimeout(() => {
            alertElement.classList.add('show');
        }, 10);

        if (duration > 0) {
            setTimeout(() => {
                this.removeAlert(alertId);
            }, duration);
        }

        return alertId;
    },

    createAlertElement(id, message, type) {
        const alertElement = document.createElement('div');
        alertElement.id = id;
        alertElement.className = `alert-glass alert-${type}`;

        alertElement.innerHTML = `
            <div class="d-flex align-items-center">
                <i class="fas fa-${this.getAlertIcon(type)} me-3"></i>
                <div class="flex-grow-1">${message}</div>
                <button class="btn btn-sm btn-glass ms-3" onclick="Utils.removeAlert('${id}')">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="alert-progress"></div>
        `;

        return alertElement;
    },

    removeAlert(alertId) {
        const alertElement = document.getElementById(alertId);
        if (alertElement) {
            alertElement.classList.add('animate__fadeOutUp');
            setTimeout(() => {
                if (alertElement.parentNode) {
                    alertElement.parentNode.removeChild(alertElement);
                }
            }, 500);
        }
    },

    getAlertIcon(type) {
        const icons = {
            'success': 'check-circle',
            'error': 'exclamation-triangle',
            'warning': 'exclamation-circle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    },

    formatNumber(num, locale = 'vi-VN') {
        return new Intl.NumberFormat(locale).format(num);
    },

    formatCurrency(num, currency = 'VND', locale = 'vi-VN') {
        return new Intl.NumberFormat(locale, {
            style: 'currency',
            currency: currency
        }).format(num);
    },

    formatPercentage(num, decimals = 2) {
        return (num * 100).toFixed(decimals) + '%';
    },

    animateValue(element, start, end, duration = 1000, formatter = null) {
        if (!element) return;

        const range = end - start;
        let startTime;

        const step = (timestamp) => {
            if (!startTime) startTime = timestamp;
            const progress = Math.min((timestamp - startTime) / duration, 1);
            const value = progress * range + start;

            const displayValue = formatter ? formatter(value) : Math.floor(value);
            element.textContent = displayValue;

            if (progress < 1) {
                requestAnimationFrame(step);
            }
        };

        requestAnimationFrame(step);
    },

    generateId() {
        return 'id_' + Math.random().toString(36).substr(2, 9);
    },

    deepClone(obj) {
        return JSON.parse(JSON.stringify(obj));
    },

    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    downloadFile(data, filename, type = 'text/plain') {
        const blob = new Blob([data], { type });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }
};

// Debounce function
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction() {
        const context = this;
        const args = arguments;
        const later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(context, args);
    };
}

// Advanced Navigation Manager
class NavigationManager {
    constructor(stateManager) {
        this.stateManager = stateManager;
        this.sections = new Map();
        this.init();
    }

    init() {
        this.setupSidebar();
        this.setupNavigation();
        this.setupKeyboardShortcuts();
    }

    setupSidebar() {
        const sidebarToggle = document.getElementById('sidebarToggle');
        const sidebar = document.getElementById('sidebar');
        const mainContent = document.getElementById('mainContent');

        if (sidebarToggle && sidebar && mainContent) {
            sidebarToggle.addEventListener('click', () => {
                const isCollapsed = sidebar.classList.contains('collapsed');

                sidebar.classList.toggle('collapsed');
                mainContent.classList.toggle('expanded');

                // Update icon
                const icon = sidebarToggle.querySelector('i');
                icon.className = isCollapsed ? 'fas fa-bars' : 'fas fa-times';

                this.stateManager.setState({ sidebarCollapsed: !isCollapsed });
            });
        }
    }

    setupNavigation() {
        document.querySelectorAll('.nav-item[data-section]').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const section = e.currentTarget.dataset.section;
                this.switchSection(section);
            });
        });
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case '1':
                        e.preventDefault();
                        this.switchSection('dashboard');
                        break;
                    case '2':
                        e.preventDefault();
                        this.switchSection('data-overview');
                        break;
                    case '3':
                        e.preventDefault();
                        this.switchSection('visualizations');
                        break;
                    case '4':
                        e.preventDefault();
                        this.switchSection('ml-analysis');
                        break;
                    case 'k':
                        e.preventDefault();
                        this.toggleSidebar();
                        break;
                }
            }
        });
    }

    switchSection(sectionName) {
        // Update active nav item
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });

        const activeNavItem = document.querySelector(`[data-section="${sectionName}"]`);
        if (activeNavItem) {
            activeNavItem.classList.add('active');
        }

        // Hide all sections
        document.querySelectorAll('.content-section').forEach(section => {
            section.style.display = 'none';
        });

        // Show selected section
        const targetSection = document.getElementById(`${sectionName}-section`);
        if (targetSection) {
            targetSection.style.display = 'block';

            // Add entrance animation
            targetSection.classList.add('animate__animated', 'animate__fadeIn');
            setTimeout(() => {
                targetSection.classList.remove('animate__animated', 'animate__fadeIn');
            }, 500);
        }

        this.stateManager.setState({ currentSection: sectionName });
        this.loadSectionContent(sectionName);
    }

    async loadSectionContent(sectionName) {
        const state = this.stateManager.getState();

        if (!state.dataLoaded && sectionName !== 'dashboard') {
            Utils.showAlert('Vui lòng tải dữ liệu trước khi xem phần này', 'warning');
            return;
        }

        try {
            switch(sectionName) {
                case 'visualizations':
                    if (!state.analysisResults.visualizations) {
                        await DataAnalysisManager.loadVisualizations();
                    }
                    break;
                case 'ml-analysis':
                    if (!state.analysisResults.mlAnalysis) {
                        await DataAnalysisManager.loadMLAnalysis();
                    }
                    break;
                case 'clustering':
                    if (!state.analysisResults.clustering) {
                        await DataAnalysisManager.loadClustering();
                    }
                    break;
                case 'statistical-tests':
                    if (!state.analysisResults.statisticalTests) {
                        await DataAnalysisManager.loadStatisticalTests();
                    }
                    break;
            }
        } catch (error) {
            console.error(`Error loading ${sectionName}:`, error);
            Utils.showAlert(`Lỗi khi tải ${sectionName}: ${error.message}`, 'error');
        }
    }

    toggleSidebar() {
        document.getElementById('sidebarToggle').click();
    }
}

// Enhanced Data Manager
class DataManager {
    constructor(stateManager) {
        this.stateManager = stateManager;
        this.cache = new Map();
    }

    async loadData() {
        Utils.showLoading(true);

        try {
            const response = await fetch(`${CONFIG.API_BASE_URL}/load_data`);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();

            if (data.status === 'success') {
                this.stateManager.setState({ dataLoaded: true });

                Utils.showAlert(`✅ ${data.message}`, 'success');

                // Update UI with data information
                this.updateQuickStats(data.basic_stats);
                this.displayDataOverview(data);

                // Enable analysis buttons
                this.enableAnalysisButtons();

                // Cache the data
                this.cache.set('loadedData', data);

                return { success: true, data };
            } else {
                throw new Error(data.message || 'Không thể tải dữ liệu');
            }
        } catch (error) {
            console.error('Data loading error:', error);
            Utils.showAlert(`❌ Lỗi: ${error.message}`, 'error');
            return { success: false, error: error.message };
        } finally {
            Utils.showLoading(false);
        }
    }

    updateQuickStats(stats) {
        const quickStatsElement = document.getElementById('quickStats');
        if (quickStatsElement) {
            quickStatsElement.style.display = 'flex';
        }

        // Animate statistics with staggered timing
        const animations = [
            { element: 'totalRecords', value: stats.total_records, delay: 200 },
            { element: 'totalColumns', value: stats.total_columns, delay: 400 },
            { element: 'missingValues', value: stats.missing_values, delay: 600 },
            { element: 'memoryUsage', value: stats.memory_usage, delay: 800, isText: true }
        ];

        animations.forEach(({ element, value, delay, isText }) => {
            setTimeout(() => {
                const el = document.getElementById(element);
                if (el) {
                    if (isText) {
                        el.textContent = value;
                    } else {
                        Utils.animateValue(el, 0, value, 1500, Utils.formatNumber);
                    }
                }
            }, delay);
        });
    }

    displayDataOverview(data) {
        const container = document.getElementById('dataOverviewContent');
        if (!container) return;

        const overviewHtml = `
            <div class="row">
                <div class="col-lg-6">
                    <div class="chart-container">
                        <h5 class="chart-title"><i class="fas fa-table"></i> Cấu trúc dữ liệu</h5>
                        <div class="table-responsive">
                            <table class="table table-dark table-hover">
                                <thead>
                                    <tr>
                                        <th><i class="fas fa-columns"></i> Cột</th>
                                        <th><i class="fas fa-tag"></i> Kiểu dữ liệu</th>
                                        <th><i class="fas fa-info-circle"></i> Mô tả</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${data.columns.map(col => `
                                        <tr>
                                            <td><code>${col}</code></td>
                                            <td><span class="badge ${this.getColumnTypeBadge(col)}">${this.getColumnType(col)}</span></td>
                                            <td><small>${this.getColumnDescription(col)}</small></td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <div class="col-lg-6">
                    <div class="chart-container">
                        <h5 class="chart-title"><i class="fas fa-eye"></i> Xem trước dữ liệu</h5>
                        <div class="table-responsive">
                            <table class="table table-dark table-striped">
                                <thead>
                                    <tr>
                                        ${data.columns.slice(0, 4).map(col => `
                                            <th style="min-width: 120px;">${col}</th>
                                        `).join('')}
                                    </tr>
                                </thead>
                                <tbody>
                                    ${data.sample_data.slice(0, 5).map(row => `
                                        <tr>
                                            ${data.columns.slice(0, 4).map(col => `
                                                <td>${this.formatCellValue(row[col])}</td>
                                            `).join('')}
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                        <div class="mt-3">
                            <small class="text-muted">
                                <i class="fas fa-info-circle"></i> 
                                Hiển thị ${Math.min(5, data.sample_data.length)} dòng đầu tiên của ${data.total_records || data.sample_data.length} dòng
                            </small>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-12">
                    <div class="chart-container">
                        <h5 class="chart-title"><i class="fas fa-chart-bar"></i> Thống kê cơ bản</h5>
                        <div class="row">
                            ${this.generateBasicStatsCards(data.basic_stats)}
                        </div>
                    </div>
                </div>
            </div>
        `;

        container.innerHTML = overviewHtml;

        // Add smooth fade-in animation
        container.querySelectorAll('.chart-container').forEach((el, index) => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(20px)';
            setTimeout(() => {
                el.style.transition = 'all 0.5s ease';
                el.style.opacity = '1';
                el.style.transform = 'translateY(0)';
            }, index * 200);
        });
    }

    generateBasicStatsCards(stats) {
        const cards = [
            {
                icon: 'fas fa-database',
                title: 'Tổng dữ liệu',
                value: Utils.formatNumber(stats.total_records),
                subtitle: 'dòng dữ liệu',
                color: 'primary'
            },
            {
                icon: 'fas fa-columns',
                title: 'Số cột',
                value: stats.total_columns,
                subtitle: 'thuộc tính',
                color: 'success'
            },
            {
                icon: 'fas fa-exclamation-triangle',
                title: 'Giá trị thiếu',
                value: Utils.formatNumber(stats.missing_values),
                subtitle: 'ô trống',
                color: 'warning'
            },
            {
                icon: 'fas fa-memory',
                title: 'Dung lượng',
                value: stats.memory_usage,
                subtitle: 'bộ nhớ',
                color: 'info'
            }
        ];

        return cards.map(card => `
            <div class="col-md-3 col-sm-6">
                <div class="stats-card stats-card-${card.color}">
                    <div class="icon"><i class="${card.icon}"></i></div>
                    <div class="value">${card.value}</div>
                    <div class="label">${card.title}</div>
                    <small class="subtitle">${card.subtitle}</small>
                </div>
            </div>
        `).join('');
    }

    getColumnType(column) {
        const numericColumns = ['Age', 'Hours_Per_Week', 'Work_Life_Balance_Score', 'Social_Isolation_Score'];
        const dateColumns = ['Survey_Date'];

        if (numericColumns.includes(column)) return 'Số';
        if (dateColumns.includes(column)) return 'Ngày';
        return 'Văn bản';
    }

    getColumnTypeBadge(column) {
        const type = this.getColumnType(column);
        switch(type) {
            case 'Số': return 'bg-primary';
            case 'Ngày': return 'bg-info';
            default: return 'bg-secondary';
        }
    }

    getColumnDescription(column) {
        const descriptions = {
            'Survey_Date': 'Ngày thực hiện khảo sát',
            'Age': 'Tuổi của người tham gia',
            'Gender': 'Giới tính',
            'Region': 'Khu vực địa lý',
            'Industry': 'Ngành nghề',
            'Job_Role': 'Vai trò công việc',
            'Work_Arrangement': 'Hình thức làm việc',
            'Hours_Per_Week': 'Số giờ làm việc mỗi tuần',
            'Mental_Health_Status': 'Tình trạng sức khỏe tinh thần',
            'Burnout_Level': 'Mức độ kiệt sức',
            'Work_Life_Balance_Score': 'Điểm cân bằng công việc-cuộc sống',
            'Physical_Health_Issues': 'Vấn đề sức khỏe thể chất',
            'Social_Isolation_Score': 'Điểm cô lập xã hội',
            'Salary_Range': 'Khoảng lương'
        };
        return descriptions[column] || 'Không có mô tả';
    }

    formatCellValue(value) {
        if (value === null || value === undefined) {
            return '<span class="text-muted">N/A</span>';
        }

        const stringValue = value.toString();
        if (stringValue.length > 25) {
            return `<span title="${stringValue}">${stringValue.substring(0, 22)}...</span>`;
        }

        return stringValue;
    }

    enableAnalysisButtons() {
        document.getElementById('startAnalysisBtn').disabled = false;
        document.getElementById('exportReportBtn').disabled = false;
    }
}

// Initialize Application
let appStateManager, navigationManager, dataManager;

document.addEventListener('DOMContentLoaded', () => {
    // Initialize managers
    appStateManager = new AppStateManager();
    navigationManager = new NavigationManager(appStateManager);
    dataManager = new DataManager(appStateManager);

    // Setup main event listeners
    setupMainEventListeners();

    // Show welcome message
    setTimeout(() => {
        Utils.showAlert(
            '🎉 Chào mừng đến với Mental Health Analytics Platform! ' +
            'Hãy bắt đầu bằng việc tải dữ liệu để khám phá các insights mạnh mẽ.',
            'info',
            10000
        );
    }, 1000);
});

function setupMainEventListeners() {
    // Load data button
    const loadDataBtn = document.getElementById('loadDataBtn');
    if (loadDataBtn) {
        loadDataBtn.addEventListener('click', async () => {
            await dataManager.loadData();
        });
    }

    // Start analysis button
    const startAnalysisBtn = document.getElementById('startAnalysisBtn');
    if (startAnalysisBtn) {
        startAnalysisBtn.addEventListener('click', async () => {
            if (typeof DataAnalysisManager !== 'undefined') {
                await DataAnalysisManager.startFullAnalysis();
            }
        });
    }

    // Export report button
    const exportReportBtn = document.getElementById('exportReportBtn');
    if (exportReportBtn) {
        exportReportBtn.addEventListener('click', () => {
            if (typeof ReportExporter !== 'undefined') {
                ReportExporter.exportReport();
            }
        });
    }

    // Feature cards
    document.querySelectorAll('.feature-card').forEach(card => {
        card.addEventListener('click', (e) => {
            const feature = e.currentTarget.dataset.feature;
            handleFeatureCardClick(feature);
        });
    });
}

function handleFeatureCardClick(feature) {
    switch(feature) {
        case 'advanced-viz':
            navigationManager.switchSection('visualizations');
            break;
        case 'ml-models':
            navigationManager.switchSection('ml-analysis');
            break;
        case 'clustering':
            navigationManager.switchSection('clustering');
            break;
        default:
            console.log('Unknown feature:', feature);
    }
}

// Export for global access
window.AppState = appStateManager;
window.Navigation = navigationManager;
window.DataManager = dataManager;
window.Utils = Utils;