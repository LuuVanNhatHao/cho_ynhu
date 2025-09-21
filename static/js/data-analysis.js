// Data Analysis Module
// Handles all analysis operations and API calls

const DataAnalysis = {
    // Start comprehensive analysis
    startFullAnalysis: async () => {
        if (!AppState.dataLoaded) {
            Utils.showAlert('Please wait for data to load', 'warning');
            return;
        }

        Utils.showLoading(true);
        Utils.showAlert('🚀 Starting comprehensive analysis...', 'info');

        try {
            const promises = [
                DataAnalysis.loadVisualizations(),
                DataAnalysis.loadMLAnalysis(),
                DataAnalysis.loadStatisticalTests(),
                WorkArrangementManager.loadAnalysis()
            ];

            await Promise.all(promises);
            Utils.showAlert('✅ Analysis complete! All results are ready.', 'success');

        } catch (error) {
            Utils.handleApiError(error);
        } finally {
            Utils.showLoading(false);
        }
    },

    // Load visualizations
    loadVisualizations: async () => {
        Utils.updateNavStatus('visualizations', 'loading');
        try {
            const response = await fetch('/api/visualizations');
            const data = await Utils.parseApiResponse(response);

            if (data.status === 'success') {
                AppState.analysisResults.visualizations = data.plots;
                VisualizationManager.renderAllPlots(data.plots);
                VisualizationManager.setupFilters();
                Utils.updateNavStatus('visualizations', 'loaded');
                return true;
            }
            return false;
        } catch (error) {
            console.error('Error loading visualizations:', error);
            Utils.updateNavStatus('visualizations', '');
            return false;
        }
    },

    // Load ML analysis
    loadMLAnalysis: async () => {
        Utils.updateNavStatus('ml-analysis', 'loading');
        try {
            const response = await fetch('/api/ml_analysis');
            const data = await Utils.parseApiResponse(response);

            if (data.status === 'success') {
                AppState.analysisResults.mlAnalysis = data;
                MLAnalysisManager.renderResults(data);
                Utils.updateNavStatus('ml-analysis', 'loaded');
                return true;
            }
            return false;
        } catch (error) {
            console.error('Error loading ML analysis:', error);
            Utils.updateNavStatus('ml-analysis', '');
            return false;
        }
    },

    // Load statistical tests
    loadStatisticalTests: async (var1 = null, var2 = null) => {
        Utils.updateNavStatus('statistical-tests', 'loading');
        try {
            let url = '/api/statistical_tests';
            const params = new URLSearchParams();
            if (var1) params.append('var1', var1);
            if (var2) params.append('var2', var2);
            if (params.toString()) url += '?' + params.toString();

            const response = await fetch(url);
            const data = await Utils.parseApiResponse(response);

            if (data.status === 'success') {
                AppState.analysisResults.statisticalTests = data.tests;
                if (var1 && var2) {
                    StatisticalTests.renderCustomTests(data.tests);
                } else {
                    StatisticalTests.renderDefaultTests(data.tests);
                }
                Utils.updateNavStatus('statistical-tests', 'loaded');
                return true;
            }
            return false;
        } catch (error) {
            console.error('Error loading statistical tests:', error);
            Utils.updateNavStatus('statistical-tests', '');
            return false;
        }
    },

    // Load recommendations
    loadRecommendations: async () => {
        Utils.updateNavStatus('recommendations', 'loading');
        if (AppState.analysisResults.mlAnalysis && AppState.analysisResults.mlAnalysis.recommendations) {
            RecommendationsManager.renderRecommendations(AppState.analysisResults.mlAnalysis.recommendations);
            Utils.updateNavStatus('recommendations', 'loaded');
            return true;
        } else {
            // Load ML analysis first if not available
            await DataAnalysis.loadMLAnalysis();
            if (AppState.analysisResults.mlAnalysis && AppState.analysisResults.mlAnalysis.recommendations) {
                RecommendationsManager.renderRecommendations(AppState.analysisResults.mlAnalysis.recommendations);
                Utils.updateNavStatus('recommendations', 'loaded');
                return true;
            }
        }
        return false;
    },

    // Perform clustering analysis
    performClustering: async (config) => {
        try {
            const url = '/api/clustering';
            const params = new URLSearchParams();

            if (config.k) params.append('k', config.k);
            if (config.features && config.features.length > 0) {
                config.features.forEach(f => params.append('features', f));
            }
            if (config.algorithm) params.append('algorithm', config.algorithm);

            const response = await fetch(url + '?' + params.toString());
            const data = await Utils.parseApiResponse(response);

            if (data.status === 'success') {
                return data.clustering_results;
            }
            return null;
        } catch (error) {
            Utils.handleApiError(error);
            return null;
        }
    },

    // Get analysis status
    getAnalysisStatus: () => {
        return {
            visualizations: !!AppState.analysisResults.visualizations,
            mlAnalysis: !!AppState.analysisResults.mlAnalysis,
            clustering: !!AppState.analysisResults.clustering,
            statisticalTests: !!AppState.analysisResults.statisticalTests,
            workArrangement: !!AppState.analysisResults.workArrangement,
            recommendations: !!AppState.analysisResults.recommendations
        };
    },

    // Clear analysis cache
    clearCache: () => {
        AppState.analysisResults = {};
        // Reset all nav status indicators
        document.querySelectorAll('.nav-item').forEach(item => {
            if (item.dataset.section !== 'dashboard') {
                item.classList.remove('loaded', 'loading');
            }
        });
    }
};