// Main Application JavaScript
// Entry point and state management

// Application State
const AppState = {
    dataLoaded: false,
    currentSection: 'dashboard',
    analysisResults: {},
    chartInstances: {},
    schemaInfo: null,
    dataFingerprint: null
};

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    // Initialize all modules
    Navigation.init();
    await DataManager.autoLoadData();

    // Setup global event listeners
    setupEventListeners();

    // Initialize feature cards
    initializeFeatureCards();
});

// Setup global event listeners
function setupEventListeners() {
    // Main control buttons
    document.getElementById('reloadDataBtn').addEventListener('click', DataManager.reloadData);
    document.getElementById('startAnalysisBtn').addEventListener('click', DataAnalysis.startFullAnalysis);
    document.getElementById('exportReportBtn').addEventListener('click', ReportExporter.exportReport);
}

// Initialize feature cards click events
function initializeFeatureCards() {
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
}

// Global error handler
window.addEventListener('error', (e) => {
    console.error('Application error:', e);
    Utils.showAlert('An unexpected error occurred. Please refresh the page.', 'error');
});

// Handle visibility change (tab switching)
document.addEventListener('visibilitychange', () => {
    if (!document.hidden && AppState.dataLoaded) {
        // Refresh data when tab becomes visible again
        console.log('Tab became visible, checking data status...');
    }
});