"""
Analytics Engine Service
Handles statistical analysis, hypothesis testing, and clustering
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, f_oneway
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import umap
import pickle
import os
from typing import Dict, List, Optional, Tuple, Any
from config import Config


class AnalyticsEngine:
    """Analytics and ML service"""

    def __init__(self, df: pd.DataFrame):
        """Initialize analytics engine"""
        self.df = df
        self.cache = {}
        self.models = {}
        self.results = {}

    def precompute_all(self):
        """Precompute heavy analytics for caching"""
        print("Pre-computing statistical tests...")
        self._precompute_hypothesis_tests()

        print("Pre-computing correlation matrix...")
        self._precompute_correlations()

        print("Pre-computing initial clusters...")
        self._precompute_clusters()

    def _precompute_hypothesis_tests(self):
        """Pre-run all hypothesis tests"""
        self.results['hypothesis'] = {
            'remote_burnout': self.test_remote_vs_burnout(),
            'gender_balance': self.test_gender_balance_score(),
            'age_mental_health': self.test_age_mental_health(),
            'industry_hours': self.test_industry_hours(),
            'salary_isolation': self.test_salary_isolation()
        }

    def _precompute_correlations(self):
        """Precompute correlation matrix"""
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        self.cache['correlation_matrix'] = self.df[num_cols].corr()

    def _precompute_clusters(self):
        """Precompute default clustering"""
        self.perform_clustering(n_clusters=5, method='kmeans')

    def update_data(self, df: pd.DataFrame):
        """Update data and clear cache"""
        self.df = df
        self.cache = {}
        self.precompute_all()

    # ========== Hypothesis Testing Methods ==========

    def test_remote_vs_burnout(self) -> Dict:
        """Test: Remote workers have lower burnout levels"""
        try:
            remote_burnout = self.df[self.df['Work_Arrangement'].str.contains('Remote', case=False, na=False)][
                'Burnout_Level']
            office_burnout = self.df[~self.df['Work_Arrangement'].str.contains('Remote', case=False, na=False)][
                'Burnout_Level']

            # Convert to numerical
            burnout_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
            remote_scores = remote_burnout.map(burnout_map).dropna()
            office_scores = office_burnout.map(burnout_map).dropna()

            # Check if we have enough data
            if len(remote_scores) < 2 or len(office_scores) < 2:
                return {
                    'test_name': 'Remote Work vs Burnout Level',
                    'hypothesis': 'Remote workers have lower burnout levels',
                    'method': 'Mann-Whitney U Test',
                    'statistic': 0.0,
                    'p_value': 1.0,
                    'effect_size': 0.0,
                    'chi2_statistic': 0.0,
                    'chi2_p_value': 1.0,
                    'significant': False,
                    'conclusion': 'Insufficient data for testing',
                    'remote_mean': float(remote_scores.mean()) if len(remote_scores) > 0 else 0.0,
                    'office_mean': float(office_scores.mean()) if len(office_scores) > 0 else 0.0,
                    'sample_sizes': {'remote': len(remote_scores), 'office': len(office_scores)}
                }

            # Mann-Whitney U test
            statistic, p_value = mannwhitneyu(remote_scores, office_scores, alternative='less')

            # Effect size (rank biserial correlation)
            n1, n2 = len(remote_scores), len(office_scores)
            effect_size = 1 - (2 * statistic) / (n1 * n2)

            # Chi-square for categorical
            contingency_table = pd.crosstab(
                self.df['Work_Arrangement'].str.contains('Remote', case=False, na=False),
                self.df['Burnout_Level']
            )
            chi2, p_chi, dof, expected = chi2_contingency(contingency_table)

            return {
                'test_name': 'Remote Work vs Burnout Level',
                'hypothesis': 'Remote workers have lower burnout levels',
                'method': 'Mann-Whitney U Test',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'effect_size': float(effect_size),
                'chi2_statistic': float(chi2),
                'chi2_p_value': float(p_chi),
                'significant': p_value < Config.SIGNIFICANCE_LEVEL,
                'conclusion': 'Remote workers show significantly lower burnout' if p_value < Config.SIGNIFICANCE_LEVEL else 'No significant difference',
                'remote_mean': float(remote_scores.mean()),
                'office_mean': float(office_scores.mean()),
                'sample_sizes': {'remote': len(remote_scores), 'office': len(office_scores)}
            }
        except Exception as e:
            print(f"Error in test_remote_vs_burnout: {e}")
            return {
                'test_name': 'Remote Work vs Burnout Level',
                'hypothesis': 'Remote workers have lower burnout levels',
                'method': 'Mann-Whitney U Test',
                'statistic': 0.0,
                'p_value': 1.0,
                'effect_size': 0.0,
                'chi2_statistic': 0.0,
                'chi2_p_value': 1.0,
                'significant': False,
                'conclusion': f'Test failed: {str(e)}',
                'remote_mean': 0.0,
                'office_mean': 0.0,
                'sample_sizes': {'remote': 0, 'office': 0}
            }

    def test_gender_balance_score(self) -> Dict:
        """Test: Gender differences in work-life balance"""
        male_scores = self.df[self.df['Gender'] == 'Male']['Work_Life_Balance_Score']
        female_scores = self.df[self.df['Gender'] == 'Female']['Work_Life_Balance_Score']

        # T-test
        statistic, p_value = stats.ttest_ind(male_scores, female_scores)

        # Cohen's d effect size
        pooled_std = np.sqrt((male_scores.std() ** 2 + female_scores.std() ** 2) / 2)
        cohens_d = (male_scores.mean() - female_scores.mean()) / pooled_std

        return {
            'test_name': 'Gender vs Work-Life Balance',
            'hypothesis': 'Gender affects work-life balance scores',
            'method': 'Independent T-Test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'effect_size': float(cohens_d),
            'significant': p_value < Config.SIGNIFICANCE_LEVEL,
            'conclusion': f"{'Significant' if p_value < Config.SIGNIFICANCE_LEVEL else 'No significant'} gender difference",
            'male_mean': float(male_scores.mean()),
            'female_mean': float(female_scores.mean()),
            'male_std': float(male_scores.std()),
            'female_std': float(female_scores.std())
        }

    def test_age_mental_health(self) -> Dict:
        """Test: Age groups and mental health status"""
        try:
            # Create age groups
            age_groups = pd.cut(self.df['Age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '50+'])

            # Contingency table
            contingency = pd.crosstab(age_groups, self.df['Mental_Health_Status'])

            # Check if contingency table is valid
            if contingency.empty or contingency.sum().sum() == 0:
                return {
                    'test_name': 'Age Groups vs Mental Health',
                    'hypothesis': 'Age affects mental health status',
                    'method': 'Chi-square Test & ANOVA',
                    'chi2_statistic': 0.0,
                    'chi2_p_value': 1.0,
                    'p_value': 1.0,
                    'cramers_v': 0.0,
                    'f_statistic': 0.0,
                    'anova_p_value': 1.0,
                    'significant': False,
                    'conclusion': 'Insufficient data for testing',
                    'contingency_table': {}
                }

            # Chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency)

            # Cramér's V for effect size
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

            # ANOVA for numerical comparison
            mental_health_map = {'Poor': 1, 'Average': 2, 'Good': 3}
            df_temp = self.df.copy()
            df_temp['Mental_Health_Numeric'] = df_temp['Mental_Health_Status'].map(mental_health_map)

            age_group_scores = []
            for group in ['<30', '30-40', '40-50', '50+']:
                scores = df_temp[age_groups == group]['Mental_Health_Numeric'].dropna()
                age_group_scores.append(scores)

            # Filter empty groups
            age_group_scores = [group for group in age_group_scores if len(group) > 0]

            if len(age_group_scores) < 2:
                f_stat, p_anova = 0.0, 1.0
            else:
                f_stat, p_anova = f_oneway(*age_group_scores)

            return {
                'test_name': 'Age Groups vs Mental Health',
                'hypothesis': 'Age affects mental health status',
                'method': 'Chi-square Test & ANOVA',
                'chi2_statistic': float(chi2),
                'chi2_p_value': float(p_value),
                'p_value': float(p_value),  # Add this key for consistency
                'cramers_v': float(cramers_v),
                'f_statistic': float(f_stat),
                'anova_p_value': float(p_anova),
                'significant': p_value < Config.SIGNIFICANCE_LEVEL,
                'conclusion': f"Age {'significantly' if p_value < Config.SIGNIFICANCE_LEVEL else 'does not significantly'} affect mental health",
                'contingency_table': contingency.to_dict()
            }
        except Exception as e:
            print(f"Error in test_age_mental_health: {e}")
            return {
                'test_name': 'Age Groups vs Mental Health',
                'hypothesis': 'Age affects mental health status',
                'method': 'Chi-square Test & ANOVA',
                'chi2_statistic': 0.0,
                'chi2_p_value': 1.0,
                'p_value': 1.0,
                'cramers_v': 0.0,
                'f_statistic': 0.0,
                'anova_p_value': 1.0,
                'significant': False,
                'conclusion': f'Test failed: {str(e)}',
                'contingency_table': {}
            }

    def test_industry_hours(self) -> Dict:
        """Test: Industry differences in working hours"""
        # Group by industry
        industry_hours = []
        industries = self.df['Industry'].value_counts().head(5).index  # Top 5 industries

        for industry in industries:
            hours = self.df[self.df['Industry'] == industry]['Hours_Per_Week'].dropna()
            industry_hours.append(hours)

        # Kruskal-Wallis test (non-parametric ANOVA)
        h_stat, p_value = kruskal(*industry_hours)

        # Effect size (eta squared)
        n = sum(len(group) for group in industry_hours)
        k = len(industry_hours)
        eta_squared = (h_stat - k + 1) / (n - k)

        # Industry statistics
        industry_stats = {}
        for i, industry in enumerate(industries):
            industry_stats[industry] = {
                'mean': float(industry_hours[i].mean()),
                'median': float(industry_hours[i].median()),
                'std': float(industry_hours[i].std()),
                'n': len(industry_hours[i])
            }

        return {
            'test_name': 'Industry vs Working Hours',
            'hypothesis': 'Industries differ in average working hours',
            'method': 'Kruskal-Wallis Test',
            'statistic': float(h_stat),
            'p_value': float(p_value),
            'eta_squared': float(eta_squared),
            'significant': p_value < Config.SIGNIFICANCE_LEVEL,
            'conclusion': f"{'Significant' if p_value < Config.SIGNIFICANCE_LEVEL else 'No significant'} industry differences",
            'industry_stats': industry_stats
        }

    def test_salary_isolation(self) -> Dict:
        """Test: Salary range and social isolation correlation"""
        try:
            # Map salary to numeric
            salary_map = {'<50k': 1, '50k-75k': 2, '75k-100k': 3, '100k-150k': 4, '>150k': 5}
            salary_numeric = self.df['Salary_Range'].map(salary_map)

            # Correlation test
            valid_idx = salary_numeric.notna() & self.df['Social_Isolation_Score'].notna()
            salary_clean = salary_numeric[valid_idx]
            isolation_clean = self.df.loc[valid_idx, 'Social_Isolation_Score']

            # Check if we have enough data
            if len(salary_clean) < 3 or len(isolation_clean) < 3:
                return {
                    'test_name': 'Salary vs Social Isolation',
                    'hypothesis': 'Higher salary correlates with social isolation',
                    'method': 'Spearman Correlation',
                    'correlation': 0.0,
                    'p_value': 1.0,
                    'significant': False,
                    'conclusion': 'Insufficient data for correlation testing',
                    'direction': 'none',
                    'strength': 'negligible',
                    'salary_groups': {}
                }

            # Spearman correlation (ordinal data)
            corr, p_value = stats.spearmanr(salary_clean, isolation_clean)

            # Group analysis
            salary_isolation = {}
            for salary, salary_num in salary_map.items():
                isolation_scores = self.df[self.df['Salary_Range'] == salary]['Social_Isolation_Score'].dropna()
                if len(isolation_scores) > 0:
                    salary_isolation[salary] = {
                        'mean': float(isolation_scores.mean()),
                        'median': float(isolation_scores.median()),
                        'std': float(isolation_scores.std()),
                        'n': len(isolation_scores)
                    }

            return {
                'test_name': 'Salary vs Social Isolation',
                'hypothesis': 'Higher salary correlates with social isolation',
                'method': 'Spearman Correlation',
                'correlation': float(corr),
                'p_value': float(p_value),
                'significant': p_value < Config.SIGNIFICANCE_LEVEL,
                'conclusion': f"{'Significant' if p_value < Config.SIGNIFICANCE_LEVEL else 'No significant'} correlation",
                'direction': 'positive' if corr > 0 else 'negative',
                'strength': self._interpret_correlation(abs(corr)),
                'salary_groups': salary_isolation
            }
        except Exception as e:
            print(f"Error in test_salary_isolation: {e}")
            return {
                'test_name': 'Salary vs Social Isolation',
                'hypothesis': 'Higher salary correlates with social isolation',
                'method': 'Spearman Correlation',
                'correlation': 0.0,
                'p_value': 1.0,
                'significant': False,
                'conclusion': f'Test failed: {str(e)}',
                'direction': 'none',
                'strength': 'negligible',
                'salary_groups': {}
            }

    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation strength"""
        if corr < 0.1:
            return 'negligible'
        elif corr < 0.3:
            return 'weak'
        elif corr < 0.5:
            return 'moderate'
        elif corr < 0.7:
            return 'strong'
        else:
            return 'very strong'

    # ========== Clustering Methods ==========

    def perform_clustering(self, n_clusters: int = 5, method: str = 'kmeans') -> Dict:
        """Perform clustering analysis"""
        # Prepare data
        features_for_clustering = [
            'Age', 'Hours_Per_Week', 'Work_Life_Balance_Score',
            'Social_Isolation_Score'
        ]

        # Add encoded categorical features
        df_cluster = self.df.copy()

        # Encode categorical
        for col in ['Work_Arrangement', 'Burnout_Level', 'Mental_Health_Status']:
            if col in df_cluster.columns:
                df_cluster[f'{col}_encoded'] = pd.Categorical(df_cluster[col]).codes
                features_for_clustering.append(f'{col}_encoded')

        # Select and scale features
        X = df_cluster[features_for_clustering].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform clustering
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=Config.RANDOM_STATE, n_init=10)
        elif method == 'gaussian_mixture':
            model = GaussianMixture(n_components=n_clusters, random_state=Config.RANDOM_STATE)
        elif method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Fit and predict
        if method == 'gaussian_mixture':
            labels = model.fit_predict(X_scaled)
        else:
            labels = model.fit_predict(X_scaled)

        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)

        # UMAP for visualization
        reducer = umap.UMAP(
            n_neighbors=Config.UMAP_N_NEIGHBORS,
            min_dist=Config.UMAP_MIN_DIST,
            random_state=Config.RANDOM_STATE
        )
        embedding = reducer.fit_transform(X_scaled)

        # Cluster profiles
        df_cluster['Cluster'] = labels
        cluster_profiles = self._generate_cluster_profiles(df_cluster)

        # Store results
        results = {
            'n_clusters': n_clusters,
            'method': method,
            'labels': labels,
            'silhouette_score': float(silhouette),
            'embedding': embedding,
            'features': features_for_clustering,
            'cluster_profiles': cluster_profiles,
            'cluster_sizes': pd.Series(labels).value_counts().to_dict(),
            'df_with_clusters': df_cluster,
            'scaler': scaler,
            'model': model
        }

        # Cache the results
        self.cache[f'clustering_{method}_{n_clusters}'] = results
        self.models[f'{method}_{n_clusters}'] = model

        return results

    def _generate_cluster_profiles(self, df: pd.DataFrame) -> Dict:
        """Generate detailed profiles for each cluster"""
        profiles = {}

        for cluster_id in df['Cluster'].unique():
            cluster_data = df[df['Cluster'] == cluster_id]

            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100,

                # Demographics
                'avg_age': float(cluster_data['Age'].mean()),
                'gender_distribution': cluster_data['Gender'].value_counts().to_dict(),
                'top_regions': cluster_data['Region'].value_counts().head(3).to_dict(),
                'top_industries': cluster_data['Industry'].value_counts().head(3).to_dict(),

                # Work characteristics
                'avg_hours': float(cluster_data['Hours_Per_Week'].mean()),
                'work_arrangement': cluster_data['Work_Arrangement'].value_counts().to_dict(),
                'job_roles': cluster_data['Job_Role'].value_counts().head(3).to_dict(),

                # Well-being metrics
                'avg_wlb_score': float(cluster_data['Work_Life_Balance_Score'].mean()),
                'avg_isolation_score': float(cluster_data['Social_Isolation_Score'].mean()),
                'burnout_distribution': cluster_data['Burnout_Level'].value_counts().to_dict(),
                'mental_health': cluster_data['Mental_Health_Status'].value_counts().to_dict(),

                # Health & compensation
                'health_issues': cluster_data['Physical_Health_Issues'].value_counts().head(3).to_dict(),
                'salary_distribution': cluster_data['Salary_Range'].value_counts().to_dict()
            }

            profiles[cluster_id] = profile

        return profiles

    def generate_personas(self, cluster_results: Dict) -> List[Dict]:
        """Generate persona descriptions for clusters"""
        personas = []
        profiles = cluster_results['cluster_profiles']

        for cluster_id, profile in profiles.items():
            # Determine key characteristics
            persona = {
                'id': cluster_id,
                'name': Config.PERSONA_NAMES.get(cluster_id, f'Persona {cluster_id}'),
                'size': profile['size'],
                'percentage': round(profile['percentage'], 1),

                # Demographics
                'typical_age': self._describe_age(profile['avg_age']),
                'primary_gender': max(profile['gender_distribution'], key=profile['gender_distribution'].get),
                'primary_region': list(profile['top_regions'].keys())[0] if profile['top_regions'] else 'Various',
                'primary_industry': list(profile['top_industries'].keys())[0] if profile[
                    'top_industries'] else 'Various',

                # Work style
                'work_hours': self._describe_hours(profile['avg_hours']),
                'work_arrangement': max(profile['work_arrangement'], key=profile['work_arrangement'].get),

                # Well-being
                'wlb_status': self._describe_wlb(profile['avg_wlb_score']),
                'isolation_level': self._describe_isolation(profile['avg_isolation_score']),
                'burnout_risk': max(profile['burnout_distribution'], key=profile['burnout_distribution'].get),
                'mental_health': max(profile['mental_health'], key=profile['mental_health'].get),

                # Description
                'description': self._generate_persona_description(cluster_id, profile)
            }

            personas.append(persona)

        return personas

    def _describe_age(self, age: float) -> str:
        """Convert age to description"""
        if age < 25:
            return "Early Career"
        elif age < 35:
            return "Young Professional"
        elif age < 45:
            return "Mid-Career"
        elif age < 55:
            return "Senior Professional"
        else:
            return "Late Career"

    def _describe_hours(self, hours: float) -> str:
        """Convert hours to description"""
        if hours < 35:
            return "Part-time"
        elif hours < 42:
            return "Standard"
        elif hours < 50:
            return "Extended"
        else:
            return "Overworked"

    def _describe_wlb(self, score: float) -> str:
        """Convert WLB score to description"""
        if score < 3:
            return "Poor Balance"
        elif score < 6:
            return "Struggling"
        elif score < 8:
            return "Moderate"
        else:
            return "Well-Balanced"

    def _describe_isolation(self, score: float) -> str:
        """Convert isolation score to description"""
        if score < 3:
            return "Well-Connected"
        elif score < 5:
            return "Moderate"
        elif score < 7:
            return "Somewhat Isolated"
        else:
            return "Highly Isolated"

    def _generate_persona_description(self, cluster_id: int, profile: Dict) -> str:
        """Generate narrative description for persona"""
        descriptions = {
            0: f"This group represents well-balanced professionals averaging {profile['avg_age']:.0f} years old, "
               f"working {profile['avg_hours']:.0f} hours per week. They maintain good work-life balance "
               f"with low burnout risk.",

            1: f"High-achieving but overworked group, typically {profile['avg_age']:.0f} years old, "
               f"working intense {profile['avg_hours']:.0f} hours weekly. They face significant burnout risk "
               f"and work-life balance challenges.",

            2: f"Remote-first workers around {profile['avg_age']:.0f} years old, embracing flexible work "
               f"arrangements. They work {profile['avg_hours']:.0f} hours weekly with varying isolation levels.",

            3: f"Entry-level professionals averaging {profile['avg_age']:.0f} years old, working "
               f"{profile['avg_hours']:.0f} hours per week. They're establishing their careers while "
               f"navigating work-life balance challenges.",

            4: f"Experienced professionals around {profile['avg_age']:.0f} years old who have optimized "
               f"their work-life balance. Working {profile['avg_hours']:.0f} hours weekly with strong "
               f"boundaries and satisfaction."
        }

        return descriptions.get(cluster_id,
                                f"This cluster represents {profile['size']} individuals ({profile['percentage']:.1f}%) "
                                f"with unique characteristics in work-life balance patterns.")

    def get_cluster_drivers(self, cluster_id: int) -> Dict:
        """Identify key drivers for a specific cluster"""
        # Get cached clustering results
        cluster_results = self.cache.get('clustering_kmeans_5')
        if not cluster_results:
            cluster_results = self.perform_clustering()

        df = cluster_results['df_with_clusters']

        # Prepare features
        feature_cols = cluster_results['features']
        X = df[feature_cols]
        y = (df['Cluster'] == cluster_id).astype(int)

        # Train Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE)
        rf.fit(X, y)

        # Get feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        # Calculate discriminative statistics
        cluster_mask = df['Cluster'] == cluster_id
        drivers = []

        for feature in importance['feature'].head(10):
            cluster_mean = df.loc[cluster_mask, feature].mean()
            overall_mean = df[feature].mean()
            diff_pct = ((cluster_mean - overall_mean) / overall_mean * 100) if overall_mean != 0 else 0

            drivers.append({
                'feature': feature.replace('_encoded', '').replace('_', ' ').title(),
                'importance': float(importance[importance['feature'] == feature]['importance'].iloc[0]),
                'cluster_mean': float(cluster_mean),
                'overall_mean': float(overall_mean),
                'difference_pct': float(diff_pct)
            })

        return {
            'cluster_id': cluster_id,
            'drivers': drivers,
            'model_score': float(rf.score(X, y))
        }

    # ========== Report Generation Methods ==========

    def generate_eda_report(self) -> Dict:
        """Generate comprehensive EDA report"""
        report = {
            'summary': {
                'total_records': len(self.df),
                'date_range': {
                    'start': str(self.df['Survey_Date'].min()) if 'Survey_Date' in self.df.columns else None,
                    'end': str(self.df['Survey_Date'].max()) if 'Survey_Date' in self.df.columns else None
                },
                'columns': len(self.df.columns),
                'missing_values': self.df.isnull().sum().sum()
            },

            'demographics': {
                'age': {
                    'mean': float(self.df['Age'].mean()),
                    'median': float(self.df['Age'].median()),
                    'std': float(self.df['Age'].std()),
                    'range': [float(self.df['Age'].min()), float(self.df['Age'].max())]
                },
                'gender': self.df['Gender'].value_counts().to_dict(),
                'regions': self.df['Region'].value_counts().to_dict(),
                'industries': self.df['Industry'].value_counts().head(10).to_dict()
            },

            'work_metrics': {
                'hours_per_week': {
                    'mean': float(self.df['Hours_Per_Week'].mean()),
                    'median': float(self.df['Hours_Per_Week'].median()),
                    'overtime_rate': float((self.df['Hours_Per_Week'] > 40).mean() * 100)
                },
                'work_arrangements': self.df['Work_Arrangement'].value_counts().to_dict(),
                'job_roles': self.df['Job_Role'].value_counts().head(10).to_dict()
            },

            'wellbeing_metrics': {
                'work_life_balance': {
                    'mean': float(self.df['Work_Life_Balance_Score'].mean()),
                    'distribution': self.df['Work_Life_Balance_Score'].value_counts().sort_index().to_dict()
                },
                'burnout': self.df['Burnout_Level'].value_counts().to_dict(),
                'mental_health': self.df['Mental_Health_Status'].value_counts().to_dict(),
                'isolation': {
                    'mean': float(self.df['Social_Isolation_Score'].mean()),
                    'high_isolation_rate': float((self.df['Social_Isolation_Score'] >= 7).mean() * 100)
                }
            },

            'health_compensation': {
                'physical_health': self.df['Physical_Health_Issues'].value_counts().head(5).to_dict(),
                'salary_ranges': self.df['Salary_Range'].value_counts().to_dict()
            },

            'correlations': {
                'top_positive': self._get_top_correlations(positive=True),
                'top_negative': self._get_top_correlations(positive=False)
            }
        }

        return report

    def _get_top_correlations(self, positive: bool = True, n: int = 5) -> List[Dict]:
        """Get top correlations from correlation matrix"""
        if 'correlation_matrix' not in self.cache:
            num_cols = self.df.select_dtypes(include=[np.number]).columns
            self.cache['correlation_matrix'] = self.df[num_cols].corr()

        corr_matrix = self.cache['correlation_matrix']

        # Get upper triangle
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Flatten and sort
        corr_list = []
        for col in upper_tri.columns:
            for row in upper_tri.index:
                val = upper_tri.loc[row, col]
                if pd.notna(val) and val != 1:
                    corr_list.append({
                        'var1': row,
                        'var2': col,
                        'correlation': float(val)
                    })

        # Sort and filter
        corr_list.sort(key=lambda x: x['correlation'], reverse=positive)

        if positive:
            return [c for c in corr_list if c['correlation'] > 0][:n]
        else:
            return [c for c in corr_list if c['correlation'] < 0][:n]

    def get_hypothesis_results(self) -> Dict:
        """Get all hypothesis test results"""
        if 'hypothesis' not in self.results:
            self._precompute_hypothesis_tests()
        return self.results['hypothesis']

    def get_hypothesis_summary(self) -> Dict:
        """Generate hypothesis testing summary"""
        results = self.get_hypothesis_results()

        summary = {
            'total_tests': len(results),
            'significant_results': sum(1 for r in results.values() if r['significant']),
            'tests': []
        }

        for key, result in results.items():
            summary['tests'].append({
                'name': result['test_name'],
                'significant': result['significant'],
                'p_value': result['p_value'],
                'conclusion': result['conclusion']
            })

        return summary

    def export_personas(self) -> Dict:
        """Export persona analysis"""
        cluster_results = self.cache.get('clustering_kmeans_5')
        if not cluster_results:
            cluster_results = self.perform_clustering()

        personas = self.generate_personas(cluster_results)

        return {
            'n_personas': len(personas),
            'clustering_method': cluster_results['method'],
            'silhouette_score': cluster_results['silhouette_score'],
            'personas': personas
        }