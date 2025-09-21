"""
Visualization Engine Service
Handles all Plotly visualizations for the dashboard
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from config import Config

class VizEngine:
    """Visualization service using Plotly"""

    def __init__(self):
        """Initialize viz engine with default settings"""
        self.theme = Config.PLOTLY_THEME
        self.colors = Config.COLOR_PALETTE
        self.height = Config.CHART_HEIGHT
        self.width = Config.CHART_WIDTH

    # ========== EDA Visualizations ==========

    def plot_demographics(self, df: pd.DataFrame) -> go.Figure:
        """Create demographics overview dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Age Distribution', 'Gender Balance',
                          'Regional Distribution', 'Industry Distribution'),
            specs=[[{'type': 'histogram'}, {'type': 'pie'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )

        # Age distribution
        fig.add_trace(
            go.Histogram(
                x=df['Age'],
                nbinsx=20,
                marker_color=self.colors[0],
                name='Age'
            ),
            row=1, col=1
        )

        # Gender pie chart
        gender_counts = df['Gender'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=gender_counts.index,
                values=gender_counts.values,
                marker=dict(colors=self.colors),
                hole=0.3
            ),
            row=1, col=2
        )

        # Regional distribution
        region_counts = df['Region'].value_counts().head(10)
        fig.add_trace(
            go.Bar(
                x=region_counts.values,
                y=region_counts.index,
                orientation='h',
                marker_color=self.colors[2],
                name='Region'
            ),
            row=2, col=1
        )

        # Industry distribution
        industry_counts = df['Industry'].value_counts().head(10)
        fig.add_trace(
            go.Bar(
                x=industry_counts.index,
                y=industry_counts.values,
                marker_color=self.colors[3],
                name='Industry',
                text=industry_counts.values,
                textposition='auto'
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Demographics Overview",
            template=self.theme
        )

        fig.update_xaxes(title_text="Age", row=1, col=1)
        fig.update_xaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Region", row=2, col=1)
        fig.update_xaxes(tickangle=-45, row=2, col=2)

        return fig

    def plot_work_arrangement(self, df: pd.DataFrame) -> go.Figure:
        """Visualize work arrangements and patterns"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Work Arrangements', 'Hours Distribution by Arrangement',
                          'Remote vs Office Metrics', 'Work Intensity'),
            specs=[[{'type': 'pie'}, {'type': 'box'}],
                   [{'type': 'bar'}, {'type': 'histogram'}]]
        )

        # Work arrangement pie
        work_counts = df['Work_Arrangement'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=work_counts.index,
                values=work_counts.values,
                marker=dict(colors=px.colors.qualitative.Set2)
            ),
            row=1, col=1
        )

        # Hours by arrangement (box plot)
        for arrangement in df['Work_Arrangement'].unique():
            hours = df[df['Work_Arrangement'] == arrangement]['Hours_Per_Week']
            fig.add_trace(
                go.Box(
                    y=hours,
                    name=arrangement,
                    showlegend=False
                ),
                row=1, col=2
            )

        # Remote vs Office comparison
        is_remote = df['Work_Arrangement'].str.contains('Remote', case=False, na=False)
        comparison_data = pd.DataFrame({
            'Remote': [
                df[is_remote]['Work_Life_Balance_Score'].mean(),
                df[is_remote]['Social_Isolation_Score'].mean(),
                (df[is_remote]['Burnout_Level'] == 'High').mean() * 10
            ],
            'Office': [
                df[~is_remote]['Work_Life_Balance_Score'].mean(),
                df[~is_remote]['Social_Isolation_Score'].mean(),
                (df[~is_remote]['Burnout_Level'] == 'High').mean() * 10
            ]
        }, index=['WLB Score', 'Isolation', 'Burnout Rate'])

        for col in comparison_data.columns:
            fig.add_trace(
                go.Bar(
                    x=comparison_data.index,
                    y=comparison_data[col],
                    name=col
                ),
                row=2, col=1
            )

        # Work intensity histogram
        fig.add_trace(
            go.Histogram(
                x=df['Hours_Per_Week'],
                nbinsx=30,
                marker_color=self.colors[4],
                name='Hours'
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            title_text="Work Arrangement Analysis",
            template=self.theme,
            showlegend=True
        )

        return fig

    def plot_burnout_by_factors(self, df: pd.DataFrame) -> go.Figure:
        """Analyze burnout levels by various factors"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Burnout Level Distribution', 'Burnout by Age Group',
                          'Burnout by Industry', 'Burnout Heatmap'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'heatmap'}]]
        )

        # Burnout distribution
        burnout_counts = df['Burnout_Level'].value_counts()
        burnout_order = ['Low', 'Medium', 'High', 'Very High']
        burnout_counts = burnout_counts.reindex(burnout_order, fill_value=0)

        fig.add_trace(
            go.Bar(
                x=burnout_counts.index,
                y=burnout_counts.values,
                marker_color=['green', 'yellow', 'orange', 'red'],
                text=burnout_counts.values,
                textposition='auto'
            ),
            row=1, col=1
        )

        # Burnout by age group
        age_groups = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100],
                           labels=['<30', '30-40', '40-50', '50+'])
        burnout_age = pd.crosstab(age_groups, df['Burnout_Level'])

        for burnout_level in burnout_order:
            if burnout_level in burnout_age.columns:
                fig.add_trace(
                    go.Bar(
                        x=burnout_age.index,
                        y=burnout_age[burnout_level],
                        name=burnout_level
                    ),
                    row=1, col=2
                )

        # Burnout by top industries
        top_industries = df['Industry'].value_counts().head(8).index
        burnout_industry = pd.crosstab(
            df[df['Industry'].isin(top_industries)]['Industry'],
            df[df['Industry'].isin(top_industries)]['Burnout_Level']
        )

        for burnout_level in burnout_order:
            if burnout_level in burnout_industry.columns:
                fig.add_trace(
                    go.Bar(
                        x=burnout_industry.index,
                        y=burnout_industry[burnout_level],
                        name=burnout_level,
                        showlegend=False
                    ),
                    row=2, col=1
                )

        # Correlation heatmap for burnout factors
        burnout_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
        df_temp = df.copy()
        df_temp['Burnout_Numeric'] = df_temp['Burnout_Level'].map(burnout_map)

        corr_factors = ['Hours_Per_Week', 'Work_Life_Balance_Score',
                       'Social_Isolation_Score', 'Age', 'Burnout_Numeric']
        corr_matrix = df_temp[corr_factors].corr()

        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            title_text="Burnout Analysis Dashboard",
            template=self.theme,
            barmode='stack'
        )

        return fig

    def plot_correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        """Create interactive correlation matrix"""
        # Select numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns

        # Calculate correlation
        corr_matrix = df[num_cols].corr()

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title="Feature Correlation Matrix",
            height=700,
            width=900,
            template=self.theme,
            xaxis=dict(tickangle=-45)
        )

        return fig

    def plot_health_issues(self, df: pd.DataFrame) -> go.Figure:
        """Visualize health issues analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Physical Health Issues', 'Mental Health Distribution',
                          'Health vs Work Hours', 'Health by Age Group'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]]
        )

        # Physical health issues
        health_issues = df['Physical_Health_Issues'].value_counts().head(10)
        fig.add_trace(
            go.Bar(
                x=health_issues.values,
                y=health_issues.index,
                orientation='h',
                marker_color=self.colors[5],
                text=health_issues.values,
                textposition='auto'
            ),
            row=1, col=1
        )

        # Mental health pie
        mental_counts = df['Mental_Health_Status'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=mental_counts.index,
                values=mental_counts.values,
                marker=dict(colors=['green', 'yellow', 'red']),
                hole=0.3
            ),
            row=1, col=2
        )

        # Health vs work hours scatter
        mental_map = {'Good': 3, 'Average': 2, 'Poor': 1}
        df_temp = df.copy()
        df_temp['Mental_Numeric'] = df_temp['Mental_Health_Status'].map(mental_map)

        fig.add_trace(
            go.Scatter(
                x=df_temp['Hours_Per_Week'],
                y=df_temp['Mental_Numeric'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=df_temp['Work_Life_Balance_Score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="WLB Score", x=1.15)
                ),
                text=df_temp['Mental_Health_Status'],
                hovertemplate='Hours: %{x}<br>Mental Health: %{text}<br>WLB: %{marker.color:.1f}'
            ),
            row=2, col=1
        )

        # Health by age group
        age_groups = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100],
                           labels=['<30', '30-40', '40-50', '50+'])
        health_age = pd.crosstab(age_groups, df['Mental_Health_Status'])

        for status in ['Good', 'Average', 'Poor']:
            if status in health_age.columns:
                fig.add_trace(
                    go.Bar(
                        x=health_age.index,
                        y=health_age[status],
                        name=status
                    ),
                    row=2, col=2
                )

        fig.update_layout(
            height=800,
            title_text="Health Analysis Dashboard",
            template=self.theme,
            showlegend=True
        )

        fig.update_xaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Hours Per Week", row=2, col=1)
        fig.update_yaxes(title_text="Mental Health (1=Poor, 3=Good)", row=2, col=1)

        return fig

    def plot_industry_comparison(self, df: pd.DataFrame) -> go.Figure:
        """Compare metrics across industries"""
        # Select top industries
        top_industries = df['Industry'].value_counts().head(8).index
        df_filtered = df[df['Industry'].isin(top_industries)]

        # Calculate metrics by industry
        industry_metrics = df_filtered.groupby('Industry').agg({
            'Hours_Per_Week': 'mean',
            'Work_Life_Balance_Score': 'mean',
            'Social_Isolation_Score': 'mean',
            'Age': 'mean'
        }).round(2)

        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Hours by Industry', 'Work-Life Balance by Industry',
                          'Social Isolation by Industry', 'Industry Comparison Radar'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'polar'}]]
        )

        # Hours bar chart
        fig.add_trace(
            go.Bar(
                x=industry_metrics.index,
                y=industry_metrics['Hours_Per_Week'],
                marker_color=self.colors[0],
                text=industry_metrics['Hours_Per_Week'],
                textposition='auto'
            ),
            row=1, col=1
        )

        # WLB bar chart
        fig.add_trace(
            go.Bar(
                x=industry_metrics.index,
                y=industry_metrics['Work_Life_Balance_Score'],
                marker_color=self.colors[1],
                text=industry_metrics['Work_Life_Balance_Score'],
                textposition='auto'
            ),
            row=1, col=2
        )

        # Isolation bar chart
        fig.add_trace(
            go.Bar(
                x=industry_metrics.index,
                y=industry_metrics['Social_Isolation_Score'],
                marker_color=self.colors[2],
                text=industry_metrics['Social_Isolation_Score'],
                textposition='auto'
            ),
            row=2, col=1
        )

        # Radar chart for top 5 industries
        top_5_industries = industry_metrics.head(5)

        categories = ['Hours/Week', 'WLB Score', 'Isolation', 'Avg Age']

        for industry in top_5_industries.index:
            values = [
                top_5_industries.loc[industry, 'Hours_Per_Week'] / 10,
                top_5_industries.loc[industry, 'Work_Life_Balance_Score'],
                top_5_industries.loc[industry, 'Social_Isolation_Score'],
                top_5_industries.loc[industry, 'Age'] / 10
            ]

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=industry[:15]  # Truncate long names
                ),
                row=2, col=2
            )

        fig.update_layout(
            height=800,
            title_text="Industry Comparison Dashboard",
            template=self.theme,
            showlegend=True,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            )
        )

        fig.update_xaxes(tickangle=-45)

        return fig

    # ========== Hypothesis Testing Visualizations ==========

    def plot_hypothesis_result(self, result: Dict, df: pd.DataFrame) -> go.Figure:
        """Visualize hypothesis test results"""
        test_name = result['test_name']

        if 'Remote Work' in test_name:
            return self._plot_remote_burnout(result, df)
        elif 'Gender' in test_name:
            return self._plot_gender_balance(result, df)
        elif 'Age Groups' in test_name:
            return self._plot_age_mental_health(result, df)
        elif 'Industry' in test_name:
            return self._plot_industry_hours(result, df)
        elif 'Salary' in test_name:
            return self._plot_salary_isolation(result, df)
        else:
            return self._plot_generic_hypothesis(result)

    def _plot_remote_burnout(self, result: Dict, df: pd.DataFrame) -> go.Figure:
        """Visualize remote vs office burnout"""
        is_remote = df['Work_Arrangement'].str.contains('Remote', case=False, na=False)

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Burnout Distribution', 'Mean Burnout Score')
        )

        # Stacked bar chart
        burnout_remote = df[is_remote]['Burnout_Level'].value_counts()
        burnout_office = df[~is_remote]['Burnout_Level'].value_counts()

        burnout_levels = ['Low', 'Medium', 'High', 'Very High']

        fig.add_trace(
            go.Bar(
                name='Remote',
                x=burnout_levels,
                y=[burnout_remote.get(level, 0) for level in burnout_levels],
                marker_color='lightblue'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                name='Office',
                x=burnout_levels,
                y=[burnout_office.get(level, 0) for level in burnout_levels],
                marker_color='coral'
            ),
            row=1, col=1
        )

        # Mean comparison
        fig.add_trace(
            go.Bar(
                x=['Remote', 'Office'],
                y=[result['remote_mean'], result['office_mean']],
                marker_color=['lightblue', 'coral'],
                text=[f"{result['remote_mean']:.2f}", f"{result['office_mean']:.2f}"],
                textposition='auto'
            ),
            row=1, col=2
        )

        # Add statistical annotation
        fig.add_annotation(
            text=f"p-value: {result['p_value']:.4f}<br>Effect size: {result['effect_size']:.3f}<br>Conclusion: {result['conclusion']}",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12),
            align="center"
        )

        fig.update_layout(
            height=500,
            title_text=f"{result['test_name']} (Mann-Whitney U Test)",
            template=self.theme,
            barmode='group'
        )

        return fig

    def _plot_gender_balance(self, result: Dict, df: pd.DataFrame) -> go.Figure:
        """Visualize gender WLB differences"""
        fig = go.Figure()

        # Box plots for each gender
        for gender in ['Male', 'Female']:
            scores = df[df['Gender'] == gender]['Work_Life_Balance_Score']
            fig.add_trace(
                go.Box(
                    y=scores,
                    name=gender,
                    boxmean='sd'
                )
            )

        # Add statistical annotation
        fig.add_annotation(
            text=f"T-statistic: {result['statistic']:.3f}<br>p-value: {result['p_value']:.4f}<br>Cohen's d: {result['effect_size']:.3f}<br>{result['conclusion']}",
            xref="paper", yref="paper",
            x=0.5, y=1.15,
            showarrow=False,
            font=dict(size=11),
            align="center"
        )

        fig.update_layout(
            height=500,
            title_text=result['test_name'],
            yaxis_title="Work-Life Balance Score",
            template=self.theme
        )

        return fig

    def _plot_age_mental_health(self, result: Dict, df: pd.DataFrame) -> go.Figure:
        """Visualize age groups vs mental health"""
        contingency = pd.DataFrame(result['contingency_table'])

        fig = go.Figure()

        # Stacked bar chart
        for mental_status in contingency.columns:
            fig.add_trace(
                go.Bar(
                    name=mental_status,
                    x=contingency.index,
                    y=contingency[mental_status]
                )
            )

        # Add chi-square results
        fig.add_annotation(
            text=f"χ² = {result['chi2_statistic']:.2f}, p = {result['chi2_p_value']:.4f}<br>Cramér's V = {result['cramers_v']:.3f}<br>{result['conclusion']}",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=11)
        )

        fig.update_layout(
            height=500,
            title_text=result['test_name'],
            xaxis_title="Age Group",
            yaxis_title="Count",
            barmode='stack',
            template=self.theme
        )

        return fig

    def _plot_industry_hours(self, result: Dict, df: pd.DataFrame) -> go.Figure:
        """Visualize industry working hours differences"""
        industry_stats = result['industry_stats']

        fig = go.Figure()

        # Box plot for each industry
        for industry in industry_stats.keys():
            hours = df[df['Industry'] == industry]['Hours_Per_Week']
            fig.add_trace(
                go.Box(
                    y=hours,
                    name=industry[:20],  # Truncate long names
                    boxmean=True
                )
            )

        # Add test results
        fig.add_annotation(
            text=f"Kruskal-Wallis H = {result['statistic']:.2f}<br>p-value = {result['p_value']:.4f}<br>η² = {result['eta_squared']:.3f}<br>{result['conclusion']}",
            xref="paper", yref="paper",
            x=0.5, y=1.1,
            showarrow=False,
            font=dict(size=11)
        )

        fig.update_layout(
            height=500,
            title_text=result['test_name'],
            yaxis_title="Hours Per Week",
            template=self.theme,
            showlegend=False
        )

        return fig

    def _plot_salary_isolation(self, result: Dict, df: pd.DataFrame) -> go.Figure:
        """Visualize salary vs isolation correlation"""
        salary_groups = result['salary_groups']

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Isolation by Salary Range', 'Correlation Scatter')
        )

        # Check if we have sufficient data
        if not salary_groups or len(salary_groups) == 0:
            # Create empty plot with message
            fig.add_annotation(
                text="Insufficient data for salary vs isolation analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14),
                row=1, col=1
            )
            fig.add_annotation(
                text="No correlation data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14),
                row=1, col=2
            )
        else:
            # Box plot by salary range
            salary_order = ['<50k', '50k-75k', '75k-100k', '100k-150k', '>150k']
            for salary in salary_order:
                if salary in salary_groups:
                    isolation = df[df['Salary_Range'] == salary]['Social_Isolation_Score'].dropna()
                    if len(isolation) > 0:  # Only add if there's data
                        fig.add_trace(
                            go.Box(
                                y=isolation,
                                name=salary,
                                showlegend=False
                            ),
                            row=1, col=1
                        )

            # Scatter plot - only if we have sufficient data
            salary_map = {'<50k': 1, '50k-75k': 2, '75k-100k': 3, '100k-150k': 4, '>150k': 5}
            df_temp = df.copy()
            df_temp['Salary_Numeric'] = df_temp['Salary_Range'].map(salary_map)

            # Clean data for scatter plot
            valid_idx = df_temp['Salary_Numeric'].notna() & df_temp['Social_Isolation_Score'].notna()
            df_clean = df_temp[valid_idx]

            if len(df_clean) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=df_clean['Salary_Numeric'],
                        y=df_clean['Social_Isolation_Score'],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=df_clean['Social_Isolation_Score'],
                            colorscale='Reds',
                            showscale=True
                        ),
                        showlegend=False
                    ),
                    row=1, col=2
                )

                # Add trend line only if we have sufficient data (at least 2 points)
                if len(df_clean) >= 2:
                    try:
                        z = np.polyfit(df_clean['Salary_Numeric'], df_clean['Social_Isolation_Score'], 1)
                        p = np.poly1d(z)
                        x_trend = np.linspace(df_clean['Salary_Numeric'].min(), df_clean['Salary_Numeric'].max(), 100)

                        fig.add_trace(
                            go.Scatter(
                                x=x_trend,
                                y=p(x_trend),
                                mode='lines',
                                line=dict(color='red', dash='dash'),
                                name='Trend',
                                showlegend=False
                            ),
                            row=1, col=2
                        )
                    except Exception as e:
                        print(f"Error adding trend line: {e}")

        # Add correlation results
        fig.add_annotation(
            text=f"Spearman ρ = {result['correlation']:.3f}, p = {result['p_value']:.4f}<br>{result['conclusion']}",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=11)
        )

        fig.update_layout(
            height=500,
            title_text=result['test_name'],
            template=self.theme
        )

        fig.update_xaxes(title_text="Salary Range", row=1, col=1)
        fig.update_xaxes(title_text="Salary Level", row=1, col=2)
        fig.update_yaxes(title_text="Social_Isolation_Score", row=1)

        return fig

    def _plot_generic_hypothesis(self, result: Dict) -> go.Figure:
        """Generic hypothesis result visualization"""
        fig = go.Figure()

        # Create a simple bar chart of the key metrics
        fig.add_trace(
            go.Bar(
                x=['Test Statistic', 'p-value', 'Effect Size'],
                y=[result.get('statistic', 0),
                   result.get('p_value', 0),
                   result.get('effect_size', 0) if result.get('effect_size') else 0],
                marker_color=self.colors[:3],
                text=[f"{result.get('statistic', 0):.3f}",
                      f"{result.get('p_value', 0):.4f}",
                      f"{result.get('effect_size', 0):.3f}" if result.get('effect_size') else "N/A"],
                textposition='auto'
            )
        )

        fig.update_layout(
            height=400,
            title_text=result['test_name'],
            template=self.theme
        )

        return fig

    # ========== Clustering Visualizations ==========

    def plot_cluster_umap(self, cluster_results: Dict) -> go.Figure:
        """Create UMAP visualization of clusters"""
        embedding = cluster_results['embedding']
        labels = cluster_results['labels']

        fig = go.Figure()

        # Plot each cluster
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            fig.add_trace(
                go.Scatter(
                    x=embedding[mask, 0],
                    y=embedding[mask, 1],
                    mode='markers',
                    name=f'Cluster {cluster_id}',
                    marker=dict(
                        size=5,
                        color=self.colors[cluster_id % len(self.colors)]
                    ),
                    text=[f'Cluster {cluster_id}'] * sum(mask),
                    hovertemplate='%{text}<br>UMAP1: %{x:.2f}<br>UMAP2: %{y:.2f}'
                )
            )

        fig.update_layout(
            height=600,
            title=f"Cluster Visualization (UMAP) - {cluster_results['method'].title()}",
            xaxis_title="UMAP Component 1",
            yaxis_title="UMAP Component 2",
            template=self.theme,
            hovermode='closest'
        )

        # Add silhouette score annotation
        fig.add_annotation(
            text=f"Silhouette Score: {cluster_results['silhouette_score']:.3f}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=12),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )

        return fig

    def plot_cluster_profile(self, cluster_results: Dict) -> go.Figure:
        """Create cluster profile comparison"""
        profiles = cluster_results['cluster_profiles']
        n_clusters = len(profiles)

        # Prepare data for radar chart
        metrics = ['avg_age', 'avg_hours', 'avg_wlb_score', 'avg_isolation_score']
        metric_labels = ['Age', 'Hours/Week', 'WLB Score', 'Isolation']

        fig = go.Figure()

        for cluster_id, profile in profiles.items():
            # Normalize values for better visualization
            values = [
                profile['avg_age'] / 60,  # Normalize age
                profile['avg_hours'] / 60,  # Normalize hours
                profile['avg_wlb_score'] / 10,  # Already 0-10
                profile['avg_isolation_score'] / 10  # Already 0-10
            ]

            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=metric_labels,
                    fill='toself',
                    name=f'Cluster {cluster_id} ({profile["size"]} people)',
                    marker=dict(color=self.colors[cluster_id % len(self.colors)])
                )
            )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            height=600,
            title="Cluster Profile Comparison",
            template=self.theme
        )

        return fig

    def plot_cluster_distribution(self, cluster_results: Dict) -> go.Figure:
        """Create cluster distribution charts"""
        profiles = cluster_results['cluster_profiles']
        df = cluster_results['df_with_clusters']

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cluster Sizes', 'Burnout by Cluster',
                          'Work Arrangement by Cluster', 'Salary by Cluster'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )

        # Cluster sizes pie chart
        sizes = cluster_results['cluster_sizes']
        fig.add_trace(
            go.Pie(
                labels=[f'Cluster {k}' for k in sizes.keys()],
                values=list(sizes.values()),
                marker=dict(colors=self.colors[:len(sizes)]),
                hole=0.3
            ),
            row=1, col=1
        )

        # Burnout by cluster
        burnout_cluster = pd.crosstab(df['Cluster'], df['Burnout_Level'])
        for burnout_level in ['Low', 'Medium', 'High', 'Very High']:
            if burnout_level in burnout_cluster.columns:
                fig.add_trace(
                    go.Bar(
                        x=burnout_cluster.index,
                        y=burnout_cluster[burnout_level],
                        name=burnout_level
                    ),
                    row=1, col=2
                )

        # Work arrangement by cluster
        work_cluster = pd.crosstab(df['Cluster'], df['Work_Arrangement'])
        for arrangement in work_cluster.columns[:3]:  # Top 3 arrangements
            fig.add_trace(
                go.Bar(
                    x=work_cluster.index,
                    y=work_cluster[arrangement],
                    name=arrangement,
                    showlegend=False
                ),
                row=2, col=1
            )

        # Salary by cluster
        salary_cluster = pd.crosstab(df['Cluster'], df['Salary_Range'])
        salary_order = ['<50k', '50k-75k', '75k-100k', '100k-150k', '>150k']
        for salary in salary_order:
            if salary in salary_cluster.columns:
                fig.add_trace(
                    go.Bar(
                        x=salary_cluster.index,
                        y=salary_cluster[salary],
                        name=salary,
                        showlegend=False
                    ),
                    row=2, col=2
                )

        fig.update_layout(
            height=800,
            title_text="Cluster Distribution Analysis",
            template=self.theme,
            barmode='stack'
        )

        fig.update_xaxes(title_text="Cluster", row=1, col=2)
        fig.update_xaxes(title_text="Cluster", row=2, col=1)
        fig.update_xaxes(title_text="Cluster", row=2, col=2)

        return fig

    def plot_feature_importance(self, drivers: Dict) -> go.Figure:
        """Visualize feature importance for cluster"""
        driver_list = drivers['drivers']

        fig = go.Figure()

        # Sort by importance
        driver_list.sort(key=lambda x: x['importance'], reverse=True)

        features = [d['feature'] for d in driver_list]
        importances = [d['importance'] for d in driver_list]
        differences = [d['difference_pct'] for d in driver_list]

        # Create bar chart
        fig.add_trace(
            go.Bar(
                x=importances,
                y=features,
                orientation='h',
                marker=dict(
                    color=differences,
                    colorscale='RdBu',
                    colorbar=dict(title="% Diff from Overall"),
                    cmin=-max(abs(min(differences)), abs(max(differences))),
                    cmax=max(abs(min(differences)), abs(max(differences)))
                ),
                text=[f"{imp:.3f} ({diff:+.1f}%)" for imp, diff in zip(importances, differences)],
                textposition='auto'
            )
        )

        fig.update_layout(
            height=500,
            title=f"Key Drivers for Cluster {drivers['cluster_id']}",
            xaxis_title="Feature Importance",
            template=self.theme
        )

        # Add model score annotation
        fig.add_annotation(
            text=f"Model Accuracy: {drivers['model_score']:.2%}",
            xref="paper", yref="paper",
            x=0.98, y=0.02,
            showarrow=False,
            font=dict(size=11),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )

        return fig

    # ========== Utility Methods ==========

    def create_summary_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """Create executive summary dashboard"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                'Work-Life Balance Distribution', 'Burnout Levels', 'Work Arrangements',
                'Age Distribution', 'Industry Distribution', 'Mental Health Status',
                'Hours Worked', 'Salary Distribution', 'Key Metrics'
            ),
            specs=[
                [{'type': 'histogram'}, {'type': 'pie'}, {'type': 'pie'}],
                [{'type': 'histogram'}, {'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'box'}, {'type': 'bar'}, {'type': 'indicator'}]
            ]
        )

        # Row 1
        fig.add_trace(
            go.Histogram(x=df['Work_Life_Balance_Score'], nbinsx=10, marker_color=self.colors[0]),
            row=1, col=1
        )

        burnout_counts = df['Burnout_Level'].value_counts()
        fig.add_trace(
            go.Pie(labels=burnout_counts.index, values=burnout_counts.values,
                  marker=dict(colors=['green', 'yellow', 'orange', 'red']), hole=0.3),
            row=1, col=2
        )

        work_counts = df['Work_Arrangement'].value_counts()
        fig.add_trace(
            go.Pie(labels=work_counts.index, values=work_counts.values, hole=0.3),
            row=1, col=3
        )

        # Row 2
        fig.add_trace(
            go.Histogram(x=df['Age'], nbinsx=20, marker_color=self.colors[1]),
            row=2, col=1
        )

        industry_top = df['Industry'].value_counts().head(5)
        fig.add_trace(
            go.Bar(x=industry_top.index, y=industry_top.values, marker_color=self.colors[2]),
            row=2, col=2
        )

        mental_counts = df['Mental_Health_Status'].value_counts()
        fig.add_trace(
            go.Pie(labels=mental_counts.index, values=mental_counts.values,
                  marker=dict(colors=['green', 'yellow', 'red']), hole=0.3),
            row=2, col=3
        )

        # Row 3
        fig.add_trace(
            go.Box(y=df['Hours_Per_Week'], marker_color=self.colors[3], boxmean='sd'),
            row=3, col=1
        )

        salary_counts = df['Salary_Range'].value_counts()
        salary_order = ['<50k', '50k-75k', '75k-100k', '100k-150k', '>150k']
        salary_counts = salary_counts.reindex(salary_order, fill_value=0)
        fig.add_trace(
            go.Bar(x=salary_counts.index, y=salary_counts.values, marker_color=self.colors[4]),
            row=3, col=2
        )

        # Key metrics indicator
        avg_wlb = df['Work_Life_Balance_Score'].mean()
        fig.add_trace(
            go.Indicator(
                mode="number+delta+gauge",
                value=avg_wlb,
                title={'text': "Avg WLB Score"},
                delta={'reference': 5, 'relative': True},
                gauge={'axis': {'range': [None, 10]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 3], 'color': "lightgray"},
                           {'range': [3, 7], 'color': "gray"},
                           {'range': [7, 10], 'color': "lightgreen"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 8}}
            ),
            row=3, col=3
        )

        fig.update_layout(
            height=1000,
            title_text="Executive Summary Dashboard",
            showlegend=False,
            template=self.theme
        )

        return fig

    def create_interactive_scatter(self, df: pd.DataFrame, x_col: str, y_col: str,
                                  color_col: str = None, size_col: str = None) -> go.Figure:
        """Create customizable scatter plot"""
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col,
            hover_data=df.columns,
            title=f"{y_col} vs {x_col}",
            template=self.theme,
            height=600
        )

        # Add trendline
        z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df[x_col].min(), df[x_col].max(), 100)

        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash')
            )
        )

        return fig

    def export_chart_as_html(self, fig: go.Figure, filename: str = "chart.html"):
        """Export Plotly figure as HTML"""
        fig.write_html(filename)
        return filename