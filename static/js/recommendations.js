// Recommendations Manager Module
// Handles actionable recommendations display and management

const RecommendationsManager = {
    // Render recommendations from analysis
    renderRecommendations: (recommendations) => {
        const container = document.getElementById('recommendationsContent');
        if (!container) return;

        if (!recommendations || recommendations.length === 0) {
            container.innerHTML = `
                <div class="alert alert-info">
                    <i class="fas fa-info-circle"></i> No recommendations available yet. 
                    Please run the ML Analysis first to generate recommendations.
                </div>
            `;
            return;
        }

        let html = '';

        recommendations.forEach((rec, index) => {
            const priorityColor = RecommendationsManager.getPriorityColor(rec.priority);
            const priorityIcon = RecommendationsManager.getPriorityIcon(rec.priority);

            html += `
                <div class="stats-card recommendation-card" data-recommendation-id="${index}">
                    <div class="d-flex justify-content-between align-items-start mb-3">
                        <h5 style="color: var(--text-primary);">
                            <i class="fas fa-${RecommendationsManager.getAreaIcon(rec.area)}"></i> 
                            ${rec.area}
                        </h5>
                        <span class="badge bg-${priorityColor}">
                            <i class="fas fa-${priorityIcon}"></i> ${rec.priority} Priority
                        </span>
                    </div>
                    
                    <div class="mb-3">
                        <strong>Recommendation:</strong> 
                        <p class="mt-2" style="color: var(--text-secondary);">${rec.recommendation}</p>
                    </div>
                    
                    <div class="mb-3">
                        <strong>Expected Impact:</strong> 
                        <p class="mt-2" style="color: var(--text-secondary);">
                            <i class="fas fa-chart-line text-success"></i> ${rec.expected_impact}
                        </p>
                    </div>
                    
                    ${rec.action_items && rec.action_items.length > 0 ? `
                        <div class="mb-3">
                            <strong>Action Items:</strong>
                            <ul class="mt-2 action-items-list">
                                ${rec.action_items.map((item, itemIndex) => `
                                    <li>
                                        <div class="form-check">
                                            <input class="form-check-input action-item-check" 
                                                   type="checkbox" 
                                                   id="action_${index}_${itemIndex}"
                                                   data-rec-id="${index}"
                                                   data-item-id="${itemIndex}">
                                            <label class="form-check-label" for="action_${index}_${itemIndex}">
                                                ${item}
                                            </label>
                                        </div>
                                    </li>
                                `).join('')}
                            </ul>
                            <div class="progress mt-2" style="height: 5px;">
                                <div class="progress-bar bg-success" 
                                     id="progress_${index}" 
                                     role="progressbar" 
                                     style="width: 0%"
                                     aria-valuenow="0" 
                                     aria-valuemin="0" 
                                     aria-valuemax="100">
                                </div>
                            </div>
                        </div>
                    ` : ''}
                    
                    ${rec.kpis && rec.kpis.length > 0 ? `
                        <div class="mb-3">
                            <strong>Key Performance Indicators:</strong>
                            <div class="d-flex flex-wrap gap-2 mt-2">
                                ${rec.kpis.map(kpi => `
                                    <span class="badge bg-secondary">
                                        <i class="fas fa-tachometer-alt"></i> ${kpi}
                                    </span>
                                `).join('')}
                            </div>
                        </div>
                    ` : ''}
                    
                    ${rec.timeline ? `
                        <div class="mb-3">
                            <strong>Timeline:</strong> 
                            <span class="badge bg-info">
                                <i class="fas fa-calendar-alt"></i> ${rec.timeline}
                            </span>
                        </div>
                    ` : ''}
                    
                    <div class="d-flex justify-content-end gap-2">
                        <button class="btn btn-sm btn-glass" onclick="RecommendationsManager.exportRecommendation(${index})">
                            <i class="fas fa-download"></i> Export
                        </button>
                        <button class="btn btn-sm btn-glass" onclick="RecommendationsManager.shareRecommendation(${index})">
                            <i class="fas fa-share"></i> Share
                        </button>
                        <button class="btn btn-sm btn-glass" onclick="RecommendationsManager.printRecommendation(${index})">
                            <i class="fas fa-print"></i> Print
                        </button>
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;

        // Store recommendations in AppState
        AppState.analysisResults.recommendations = recommendations;

        // Setup action item tracking
        RecommendationsManager.setupActionItemTracking();

        // Load saved progress if exists
        RecommendationsManager.loadSavedProgress();
    },

    // Get priority color based on level
    getPriorityColor: (priority) => {
        const colors = {
            'High': 'danger',
            'Medium': 'warning',
            'Low': 'info'
        };
        return colors[priority] || 'secondary';
    },

    // Get priority icon
    getPriorityIcon: (priority) => {
        const icons = {
            'High': 'exclamation-triangle',
            'Medium': 'exclamation-circle',
            'Low': 'info-circle'
        };
        return icons[priority] || 'question-circle';
    },

    // Get area icon
    getAreaIcon: (area) => {
        const icons = {
            'Work-Life Balance': 'balance-scale',
            'Social Interaction': 'users',
            'Workload Management': 'tasks',
            'Mental Health': 'brain',
            'Productivity': 'chart-line',
            'Team Building': 'people-carry'
        };
        return icons[area] || 'clipboard-list';
    },

    // Setup action item tracking
    setupActionItemTracking: () => {
        document.querySelectorAll('.action-item-check').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const recId = e.target.dataset.recId;
                const itemId = e.target.dataset.itemId;
                RecommendationsManager.updateProgress(recId);
                RecommendationsManager.saveProgress(recId, itemId, e.target.checked);
            });
        });
    },

    // Update progress bar for a recommendation
    updateProgress: (recId) => {
        const checkboxes = document.querySelectorAll(`[data-rec-id="${recId}"]`);
        const checked = document.querySelectorAll(`[data-rec-id="${recId}"]:checked`);
        const progressBar = document.getElementById(`progress_${recId}`);

        if (progressBar && checkboxes.length > 0) {
            const percentage = (checked.length / checkboxes.length) * 100;
            progressBar.style.width = `${percentage}%`;
            progressBar.setAttribute('aria-valuenow', percentage);

            // Add success animation when complete
            if (percentage === 100) {
                const card = document.querySelector(`[data-recommendation-id="${recId}"]`);
                if (card) {
                    card.classList.add('recommendation-complete');
                    Utils.showAlert('🎉 Recommendation completed!', 'success', 3000);
                }
            }
        }
    },

    // Save progress to localStorage
    saveProgress: (recId, itemId, checked) => {
        const progressKey = `recommendation_progress_${AppState.dataFingerprint}`;
        let progress = JSON.parse(localStorage.getItem(progressKey) || '{}');

        if (!progress[recId]) {
            progress[recId] = {};
        }

        progress[recId][itemId] = checked;
        localStorage.setItem(progressKey, JSON.stringify(progress));
    },

    // Load saved progress from localStorage
    loadSavedProgress: () => {
        const progressKey = `recommendation_progress_${AppState.dataFingerprint}`;
        const progress = JSON.parse(localStorage.getItem(progressKey) || '{}');

        Object.entries(progress).forEach(([recId, items]) => {
            Object.entries(items).forEach(([itemId, checked]) => {
                const checkbox = document.getElementById(`action_${recId}_${itemId}`);
                if (checkbox && checked) {
                    checkbox.checked = true;
                }
            });
            RecommendationsManager.updateProgress(recId);
        });
    },

    // Export single recommendation
    exportRecommendation: (index) => {
        const recommendations = AppState.analysisResults.recommendations;
        if (!recommendations || !recommendations[index]) {
            Utils.showAlert('Recommendation not found', 'error');
            return;
        }

        const rec = recommendations[index];
        const data = {
            timestamp: new Date().toISOString(),
            recommendation: rec
        };

        Utils.downloadJSON(data, `recommendation_${rec.area.replace(/\s+/g, '_')}_${Date.now()}.json`);
        Utils.showAlert('Recommendation exported successfully!', 'success');
    },

    // Export all recommendations
    exportAllRecommendations: () => {
        const recommendations = AppState.analysisResults.recommendations;
        if (!recommendations || recommendations.length === 0) {
            Utils.showAlert('No recommendations to export', 'warning');
            return;
        }

        const data = {
            timestamp: new Date().toISOString(),
            total_recommendations: recommendations.length,
            recommendations: recommendations
        };

        Utils.downloadJSON(data, `all_recommendations_${Date.now()}.json`);
        Utils.showAlert('All recommendations exported successfully!', 'success');
    },

    // Share recommendation (copy to clipboard)
    shareRecommendation: (index) => {
        const recommendations = AppState.analysisResults.recommendations;
        if (!recommendations || !recommendations[index]) {
            Utils.showAlert('Recommendation not found', 'error');
            return;
        }

        const rec = recommendations[index];
        const shareText = `
Recommendation: ${rec.area}
Priority: ${rec.priority}

${rec.recommendation}

Expected Impact: ${rec.expected_impact}

Action Items:
${rec.action_items ? rec.action_items.map((item, i) => `${i + 1}. ${item}`).join('\n') : 'N/A'}

Timeline: ${rec.timeline || 'N/A'}
        `.trim();

        if (navigator.clipboard) {
            navigator.clipboard.writeText(shareText).then(() => {
                Utils.showAlert('Recommendation copied to clipboard!', 'success');
            }).catch(() => {
                Utils.showAlert('Failed to copy to clipboard', 'error');
            });
        } else {
            Utils.showAlert('Clipboard not supported in this browser', 'warning');
        }
    },

    // Print single recommendation
    printRecommendation: (index) => {
        const recommendations = AppState.analysisResults.recommendations;
        if (!recommendations || !recommendations[index]) {
            Utils.showAlert('Recommendation not found', 'error');
            return;
        }

        const rec = recommendations[index];
        const printWindow = window.open('', '_blank');
        const printContent = `
            <!DOCTYPE html>
            <html>
            <head>
                <title>Recommendation: ${rec.area}</title>
                <style>
                    body { font-family: Arial, sans-serif; padding: 20px; }
                    h1 { color: #2563eb; }
                    .priority { 
                        display: inline-block;
                        padding: 5px 10px;
                        background: ${rec.priority === 'High' ? '#ef4444' : rec.priority === 'Medium' ? '#f59e0b' : '#06b6d4'};
                        color: white;
                        border-radius: 5px;
                    }
                    ul { margin: 10px 0; padding-left: 20px; }
                    .section { margin: 20px 0; }
                </style>
            </head>
            <body>
                <h1>${rec.area}</h1>
                <div class="priority">Priority: ${rec.priority}</div>
                
                <div class="section">
                    <h3>Recommendation</h3>
                    <p>${rec.recommendation}</p>
                </div>
                
                <div class="section">
                    <h3>Expected Impact</h3>
                    <p>${rec.expected_impact}</p>
                </div>
                
                ${rec.action_items ? `
                    <div class="section">
                        <h3>Action Items</h3>
                        <ul>
                            ${rec.action_items.map(item => `<li>${item}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
                
                ${rec.timeline ? `
                    <div class="section">
                        <h3>Timeline</h3>
                        <p>${rec.timeline}</p>
                    </div>
                ` : ''}
                
                <div class="section">
                    <small>Generated on: ${new Date().toLocaleString()}</small>
                </div>
            </body>
            </html>
        `;

        printWindow.document.write(printContent);
        printWindow.document.close();
        printWindow.print();
    }
};