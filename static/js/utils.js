// Utility Functions Module
// Common helper functions used throughout the application

const Utils = {
    // Show/hide loading overlay
    showLoading: (show = true) => {
        document.getElementById('loadingContainer').style.display = show ? 'flex' : 'none';
    },

    // Show auto-loading status indicator
    showAutoLoadingStatus: (status, message) => {
        const statusElement = document.getElementById('autoLoadingStatus');
        const statusIndicator = statusElement.querySelector('.status-indicator');
        const statusText = document.getElementById('statusText');

        statusElement.style.display = 'flex';
        statusText.textContent = message;

        statusIndicator.className = 'status-indicator ' + status;

        if (status === 'success' || status === 'error') {
            setTimeout(() => {
                statusElement.style.display = 'none';
            }, 5000);
        }
    },

    // Show alert messages
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

    // Get alert icon based on type
    getAlertIcon: (type) => {
        const icons = {
            'success': 'check-circle',
            'error': 'exclamation-triangle',
            'warning': 'exclamation-circle',
            'info': 'info-circle'
        };
        return icons[type] || 'info-circle';
    },

    // Format numbers with thousand separators
    formatNumber: (num) => {
        return new Intl.NumberFormat().format(num);
    },

    // Animate number counting
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
    },

    // Update navigation status indicator
    updateNavStatus: (section, status) => {
        const navItem = document.querySelector(`[data-section="${section}"]`);
        if (navItem) {
            navItem.classList.remove('loading', 'loaded');
            if (status) {
                navItem.classList.add(status);
            }
        }
    },

    // Debounce function for performance
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // Throttle function for performance
    throttle: (func, limit) => {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    // Deep clone object
    deepClone: (obj) => {
        return JSON.parse(JSON.stringify(obj));
    },

    // Check if element is in viewport
    isInViewport: (element) => {
        const rect = element.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.right <= (window.innerWidth || document.documentElement.clientWidth)
        );
    },

    // Download data as JSON file
    downloadJSON: (data, filename) => {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    },

    // Generate unique ID
    generateId: () => {
        return 'id_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    },

    // Parse API response
    parseApiResponse: async (response) => {
        try {
            const data = await response.json();
            if (data.status === 'error') {
                throw new Error(data.message || 'API error occurred');
            }
            return data;
        } catch (error) {
            throw new Error('Failed to parse API response: ' + error.message);
        }
    },

    // Handle API errors
    handleApiError: (error) => {
        console.error('API Error:', error);
        Utils.showAlert(
            `Error: ${error.message || 'An unexpected error occurred'}`,
            'error'
        );
    }
};