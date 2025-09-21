// Navigation Management Module
// Handles sidebar navigation and section switching

const Navigation = {
    // Initialize navigation
    init: () => {
        // Setup sidebar toggle
        document.getElementById('sidebarToggle').addEventListener('click', () => {
            const sidebar = document.getElementById('sidebar');
            const mainContent = document.getElementById('mainContent');

            sidebar.classList.toggle('collapsed');
            mainContent.classList.toggle('expanded');
        });

        // Setup navigation items
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const section = e.currentTarget.dataset.section;
                Navigation.switchSection(section);
            });
        });

        // Handle browser back/forward buttons
        window.addEventListener('popstate', (e) => {
            if (e.state && e.state.section) {
                Navigation.switchSection(e.state.section, false);
            }
        });
    },

    // Switch to different section
    switchSection: (section, updateHistory = true) => {
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
        const targetSection = document.getElementById(`${section}-section`);
        if (targetSection) {
            targetSection.style.display = 'block';
        }

        // Update app state
        AppState.currentSection = section;

        // Update browser history
        if (updateHistory) {
            history.pushState(
                { section: section },
                '',
                `#${section}`
            );
        }

        // Load section content if not already loaded
        Navigation.loadSectionContent(section);
    },

    // Load content for specific section
    loadSectionContent: async (section) => {
        if (!AppState.dataLoaded) return;

        switch (section) {
            case 'visualizations':
                if (!AppState.analysisResults.visualizations) {
                    await DataAnalysis.loadVisualizations();
                }
                break;

            case 'ml-analysis':
                if (!AppState.analysisResults.mlAnalysis) {
                    await DataAnalysis.loadMLAnalysis();
                }
                break;

            case 'clustering':
                if (!AppState.analysisResults.clustering) {
                    ClusteringManager.setupControls();
                }
                break;

            case 'statistical-tests':
                if (!AppState.analysisResults.statisticalTests) {
                    StatisticalTests.setupControls();
                }
                break;

            case 'work-arrangement':
                if (!AppState.analysisResults.workArrangement) {
                    WorkArrangementManager.loadAnalysis();
                }
                break;

            case 'recommendations':
                if (!AppState.analysisResults.recommendations) {
                    await DataAnalysis.loadRecommendations();
                }
                break;
        }
    },

    // Check if section is loaded
    isSectionLoaded: (section) => {
        switch (section) {
            case 'dashboard':
                return AppState.dataLoaded;
            case 'data-overview':
                return AppState.dataLoaded;
            case 'visualizations':
                return !!AppState.analysisResults.visualizations;
            case 'ml-analysis':
                return !!AppState.analysisResults.mlAnalysis;
            case 'clustering':
                return !!AppState.analysisResults.clustering;
            case 'statistical-tests':
                return !!AppState.analysisResults.statisticalTests;
            case 'work-arrangement':
                return !!AppState.analysisResults.workArrangement;
            case 'recommendations':
                return !!AppState.analysisResults.recommendations;
            default:
                return false;
        }
    },

    // Show section loading state
    showSectionLoading: (section) => {
        const targetSection = document.getElementById(`${section}-section`);
        if (targetSection) {
            const containers = targetSection.querySelectorAll('.chart-container');
            containers.forEach(container => {
                const content = container.querySelector('[id$="Content"]');
                if (content && !content.querySelector('.skeleton')) {
                    content.innerHTML = `
                        <div class="skeleton skeleton-chart"></div>
                    `;
                }
            });
        }
    },

    // Hide section loading state
    hideSectionLoading: (section) => {
        const targetSection = document.getElementById(`${section}-section`);
        if (targetSection) {
            const skeletons = targetSection.querySelectorAll('.skeleton');
            skeletons.forEach(skeleton => {
                if (skeleton.parentElement.children.length === 1) {
                    skeleton.remove();
                }
            });
        }
    },

    // Handle initial URL hash
    handleInitialHash: () => {
        const hash = window.location.hash.replace('#', '');
        if (hash && document.querySelector(`[data-section="${hash}"]`)) {
            Navigation.switchSection(hash, false);
        }
    }
};

// Handle initial hash on page load
window.addEventListener('load', () => {
    Navigation.handleInitialHash();
});