// Report Exporter Module
// Handles comprehensive report generation and export

const ReportExporter = {
    // Main export function
    exportReport: () => {
        const reportData = ReportExporter.generateReportData();
        const format = ReportExporter.getExportFormat();

        switch (format) {
            case 'html':
                ReportExporter.exportAsHTML(reportData);
                break;
            case 'pdf':
                ReportExporter.exportAsPDF(reportData);
                break;
            case 'json':
                ReportExporter.exportAsJSON(reportData);
                break;
            case 'csv':
                ReportExporter.exportAsCSV(reportData);
                break;
            default:
                ReportExporter.exportAsHTML(reportData);
        }
    },

    // Get user's preferred export format
    getExportFormat: () => {
        // You can add a dialog or dropdown to let users choose
        // For now, default to HTML
        return 'html';
    },

    // Generate comprehensive report data
    generateReportData: () => {
        const timestamp = new Date();
        const analysisStatus = DataAnalysis.getAnalysisStatus();

        return {
            metadata: {
                generated_at: timestamp.toISOString(),
                generated_by: 'Mental Health Analytics Platform',
                data_fingerprint: AppState.dataFingerprint,
                data_loaded_at: AppState.dataLoadedAt
            },
            summary: {
                total_records: AppState.basicStats?.total_records || 0,
                total_columns: AppState.basicStats?.total_columns || 0,
                missing_values: AppState.basicStats?.missing_values || 0,
                memory_usage: AppState.basicStats?.memory_usage || 'N/A'
            },
            analysis_status: analysisStatus,
            results: {
                ml_analysis: ReportExporter.extractMLResults(),
                clustering: ReportExporter.extractClusteringResults(),
                statistical_tests: ReportExporter.extractStatisticalResults(),
                work_arrangement: ReportExporter.extractWorkArrangementResults(),
                recommendations: AppState.analysisResults.recommendations || []
            }
        };
    },

    // Extract ML analysis results
    extractMLResults: () => {
        const mlData = AppState.analysisResults.mlAnalysis;
        if (!mlData) return null;

        return {
            best_model: mlData.best_model,
            models_performance: mlData.models_performance,
            feature_importance: mlData.feature_importance,
            accuracy: mlData.models_performance?.[mlData.best_model]?.accuracy
        };
    },

    // Extract clustering results
    extractClusteringResults: () => {
        const clusterData = AppState.analysisResults.clustering;
        if (!clusterData) return null;

        return {
            optimal_k: clusterData.optimal_k,
            algorithm: clusterData.algorithm,
            cluster_summary: clusterData.cluster_summary,
            silhouette_scores: clusterData.silhouette_scores
        };
    },

    // Extract statistical test results
    extractStatisticalResults: () => {
        const statsData = AppState.analysisResults.statisticalTests;
        if (!statsData) return null;

        return Object.entries(statsData).map(([testName, result]) => ({
            test_name: result.test_name,
            p_value: result.p_value,
            statistic: result.statistic,
            interpretation: result.interpretation,
            significant: result.p_value < (result.alpha || 0.05)
        }));
    },

    // Extract work arrangement results
    extractWorkArrangementResults: () => {
        const workData = AppState.analysisResults.workArrangement;
        if (!workData) return null;

        return {
            mental_health_distribution: workData.mental_health_distribution,
            stress_burnout_analysis: workData.stress_burnout_analysis,
            demographic_analysis: workData.demographic_analysis
        };
    },

    // Export as HTML
    exportAsHTML: (reportData) => {
        const htmlContent = ReportExporter.generateHTMLReport(reportData);
        ReportExporter.downloadFile(htmlContent, 'text/html', `mental_health_report_${Date.now()}.html`);
        Utils.showAlert('Report exported successfully as HTML!', 'success');
    },

    // Generate HTML report
    generateHTMLReport: (data) => {
        return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mental Health Analytics Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #334155;
            background: #f8fafc;
            padding: 40px 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2563eb, #06b6d4);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        .content {
            padding: 40px;
        }
        .section {
            margin-bottom: 40px;
            padding: 25px;
            background: #f8fafc;
            border-radius: 8px;
            border-left: 4px solid #2563eb;
        }
        .section h2 {
            color: #2563eb;
            margin-bottom: 20px;
            font-size: 1.8rem;
        }
        .section h3 {
            color: #475569;
            margin: 20px 0 10px;
            font-size: 1.3rem;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-card .value {
            font-size: 2rem;
            font-weight: bold;
            color: #2563eb;
        }
        .stat-card .label {
            color: #64748b;
            margin-top: 5px;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 2px;
        }
        .badge.success { background: #10b981; color: white; }
        .badge.warning { background: #f59e0b; color: white; }
        .badge.danger { background: #ef4444; color: white; }
        .badge.info { background: #06b6d4; color: white; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }
        th {
            background: #f1f5f9;
            font-weight: 600;
            color: #475569;
        }
        .footer {
            background: #1e293b;
            color: white;
            padding: 30px;
            text-align: center;
        }
        .recommendation {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        ul {
            margin-left: 20px;
            margin-top: 10px;
        }
        li {
            margin: 5px 0;
        }
        @media print {
            body { padding: 0; }
            .container { box-shadow: none; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Mental Health Analytics Report</h1>
            <p>Generated on ${new Date(data.metadata.generated_at).toLocaleString()}</p>
            <p>Data Fingerprint: ${data.metadata.data_fingerprint || 'N/A'}</p>
        </div>
        
        <div class="content">
            <!-- Executive Summary -->
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="value">${data.summary.total_records.toLocaleString()}</div>
                        <div class="label">Total Records</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">${data.summary.total_columns}</div>
                        <div class="label">Data Columns</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">${data.summary.missing_values.toLocaleString()}</div>
                        <div class="label">Missing Values</div>
                    </div>
                    <div class="stat-card">
                        <div class="value">${data.summary.memory_usage}</div>
                        <div class="label">Memory Usage</div>
                    </div>
                </div>
            </div>
            
            <!-- Analysis Status -->
            <div class="section">
                <h2>Analysis Status</h2>
                <table>
                    <tr>
                        <th>Analysis Type</th>
                        <th>Status</th>
                    </tr>
                    ${Object.entries(data.analysis_status).map(([key, value]) => `
                        <tr>
                            <td>${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</td>
                            <td><span class="badge ${value ? 'success' : 'info'}">${value ? 'Completed' : 'Pending'}</span></td>
                        </tr>
                    `).join('')}
                </table>
            </div>
            
            <!-- ML Analysis Results -->
            ${data.results.ml_analysis ? `
            <div class="section">
                <h2>Machine Learning Analysis</h2>
                <h3>Best Model: ${data.results.ml_analysis.best_model}</h3>
                <p>Accuracy: ${(data.results.ml_analysis.accuracy * 100).toFixed(2)}%</p>
                
                <h3>Model Performance Comparison</h3>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Accuracy</th>
                        <th>CV Mean</th>
                        <th>CV Std</th>
                    </tr>
                    ${Object.entries(data.results.ml_analysis.models_performance).map(([model, metrics]) => `
                        <tr>
                            <td>${model}</td>
                            <td>${(metrics.accuracy * 100).toFixed(2)}%</td>
                            <td>${(metrics.cv_mean * 100).toFixed(2)}%</td>
                            <td>±${(metrics.cv_std * 100).toFixed(2)}%</td>
                        </tr>
                    `).join('')}
                </table>
            </div>
            ` : ''}
            
            <!-- Recommendations -->
            ${data.results.recommendations && data.results.recommendations.length > 0 ? `
            <div class="section">
                <h2>Recommendations</h2>
                ${data.results.recommendations.map(rec => `
                    <div class="recommendation">
                        <h3>${rec.area}</h3>
                        <p><strong>Priority:</strong> <span class="badge ${rec.priority === 'High' ? 'danger' : rec.priority === 'Medium' ? 'warning' : 'info'}">${rec.priority}</span></p>
                        <p><strong>Recommendation:</strong> ${rec.recommendation}</p>
                        <p><strong>Expected Impact:</strong> ${rec.expected_impact}</p>
                        ${rec.action_items ? `
                            <p><strong>Action Items:</strong></p>
                            <ul>
                                ${rec.action_items.map(item => `<li>${item}</li>`).join('')}
                            </ul>
                        ` : ''}
                        ${rec.timeline ? `<p><strong>Timeline:</strong> ${rec.timeline}</p>` : ''}
                    </div>
                `).join('')}
            </div>
            ` : ''}
        </div>
        
        <div class="footer">
            <p>© 2024 Mental Health Analytics Platform</p>
            <p>This report contains confidential information and should be handled accordingly.</p>
        </div>
    </div>
</body>
</html>`;
    },

    // Export as JSON
    exportAsJSON: (reportData) => {
        Utils.downloadJSON(reportData, `mental_health_report_${Date.now()}.json`);
        Utils.showAlert('Report exported successfully as JSON!', 'success');
    },

    // Export as CSV (summary only)
    exportAsCSV: (reportData) => {
        const csvContent = ReportExporter.generateCSVReport(reportData);
        ReportExporter.downloadFile(csvContent, 'text/csv', `mental_health_report_${Date.now()}.csv`);
        Utils.showAlert('Report summary exported as CSV!', 'success');
    },

    // Generate CSV report
    generateCSVReport: (data) => {
        const rows = [
            ['Mental Health Analytics Report'],
            ['Generated', new Date(data.metadata.generated_at).toLocaleString()],
            [],
            ['Summary Statistics'],
            ['Metric', 'Value'],
            ['Total Records', data.summary.total_records],
            ['Total Columns', data.summary.total_columns],
            ['Missing Values', data.summary.missing_values],
            ['Memory Usage', data.summary.memory_usage],
            []
        ];

        if (data.results.ml_analysis) {
            rows.push(
                ['Machine Learning Results'],
                ['Best Model', data.results.ml_analysis.best_model],
                ['Accuracy', (data.results.ml_analysis.accuracy * 100).toFixed(2) + '%']
            );
        }

        return rows.map(row => row.map(cell =>
            typeof cell === 'string' && cell.includes(',') ? `"${cell}"` : cell
        ).join(',')).join('\n');
    },

    // Export as PDF (requires external library or browser print)
    exportAsPDF: (reportData) => {
        const htmlContent = ReportExporter.generateHTMLReport(reportData);
        const printWindow = window.open('', '_blank');
        printWindow.document.write(htmlContent);
        printWindow.document.close();

        // Trigger print dialog (user can save as PDF)
        setTimeout(() => {
            printWindow.print();
        }, 500);

        Utils.showAlert('Print dialog opened. Save as PDF from the print menu.', 'info');
    },

    // Download file helper
    downloadFile: (content, mimeType, filename) => {
        const blob = new Blob([content], { type: mimeType });
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