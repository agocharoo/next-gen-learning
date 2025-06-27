# filepath: /Users/m0c013f/Documents/next-gen-learning/Statistic/experiments.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import os
warnings.filterwarnings('ignore')

# Import our statistics module
from statistics_101 import data_generator, data_collection

class CLTExperiment:
    """
    Central Limit Theorem Experiment Class
    
    This class runs comprehensive experiments to test how sample size affects
    the normality of sample means across different population distributions.
    """
    
    def __init__(self):
        self.population_size = 100000
        self.num_samples = 1000
        self.sample_sizes = [5, 10, 15, 20, 25, 30, 50, 100]
        self.distributions = ['normal', 'bernoulli', 'poisson', 'exponential', 'beta']
        
    def run_single_experiment(self, distribution_type, sample_size):
        """
        Run a single experiment for given distribution and sample size
        
        Returns:
        dict: Results including mean, std, skewness, kurtosis, and p-values
        """
        # Generate population data
        population = data_generator(distribution_type=distribution_type, 
                                  data_points=self.population_size)
        
        # Collect sample means
        sample_means = data_collection(sample_size=sample_size, 
                                     num_samples=self.num_samples, 
                                     population_data=population)
        
        # Calculate statistics
        mean_val = np.mean(sample_means)
        std_val = np.std(sample_means)
        skewness = stats.skew(sample_means)
        kurtosis = stats.kurtosis(sample_means)
        
        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(sample_means)
        ks_stat, ks_p = stats.kstest(sample_means, 'norm', 
                                    args=(mean_val, std_val))
        
        return {
            'distribution': distribution_type,
            'sample_size': sample_size,
            'mean': mean_val,
            'std': std_val,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'shapiro_statistic': shapiro_stat,
            'shapiro_pvalue': shapiro_p,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_p,
            'sample_means': sample_means
        }
    
    def run_all_experiments(self):
        """
        Run experiments for all combinations of distributions and sample sizes
        
        Returns:
        pd.DataFrame: Results table with all statistics
        """
        results = []
        
        print("🔬 RUNNING COMPREHENSIVE CLT EXPERIMENTS")
        print("="*60)
        print(f"Population size: {self.population_size:,}")
        print(f"Number of samples: {self.num_samples:,}")
        print(f"Sample sizes: {self.sample_sizes}")
        print(f"Distributions: {self.distributions}")
        print("="*60)
        
        for dist in self.distributions:
            print(f"\n🧪 Testing {dist.upper()} distribution...")
            
            for sample_size in self.sample_sizes:
                print(f"  ⚙️ Sample size: {sample_size}", end="")
                
                result = self.run_single_experiment(dist, sample_size)
                results.append(result)
                
                # Quick feedback
                if result['shapiro_pvalue'] > 0.05:
                    print(" ✅ Normal")
                else:
                    print(" ❌ Not Normal")
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Round numerical columns for better display
        numerical_cols = ['mean', 'std', 'skewness', 'kurtosis', 
                         'shapiro_statistic', 'shapiro_pvalue', 
                         'ks_statistic', 'ks_pvalue']
        df[numerical_cols] = df[numerical_cols].round(4)
        
        return df
    
    def create_summary_table(self, results_df):
        """
        Create a formatted summary table for each distribution
        """
        print("\n" + "="*100)
        print("📊 COMPREHENSIVE RESULTS SUMMARY")
        print("="*100)
        
        for dist in self.distributions:
            dist_data = results_df[results_df['distribution'] == dist].copy()
            
            print(f"\n🔍 {dist.upper()} DISTRIBUTION")
            print("-" * 80)
            
            # Create a clean table
            display_df = dist_data[['sample_size', 'mean', 'std', 'skewness', 
                                   'kurtosis', 'shapiro_pvalue', 'ks_pvalue']].copy()
            
            # Add normality verdict
            display_df['normality_verdict'] = display_df['shapiro_pvalue'].apply(
                lambda x: '✅ Normal' if x > 0.05 else '❌ Not Normal'
            )
            
            # Rename columns for better display
            display_df.columns = ['Sample Size', 'Mean', 'Std Dev', 'Skewness', 
                                'Kurtosis', 'Shapiro p-val', 'KS p-val', 'Verdict']
            
            print(display_df.to_string(index=False))
            
            # Summary statistics
            normal_count = sum(dist_data['shapiro_pvalue'] > 0.05)
            print(f"\n📈 Summary: {normal_count}/{len(dist_data)} sample sizes show normal distribution")
            
            # Find threshold
            threshold_sample = dist_data[dist_data['shapiro_pvalue'] > 0.05]['sample_size'].min()
            if pd.notna(threshold_sample):
                print(f"🎯 Normality threshold: Sample size ≥ {threshold_sample}")
            else:
                print("🎯 Normality threshold: Not achieved in tested range")
    
    def create_visualizations(self, results_df):
        """
        Create comprehensive visualizations
        """
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. P-value heatmap
        self._plot_pvalue_heatmap(results_df)
        
        # 2. Distribution plots for key sample sizes
        self._plot_distribution_comparison(results_df)
        
        # 3. Statistics evolution
        self._plot_statistics_evolution(results_df)
    
    def create_pdf_report(self, results_df, filename=None):
        """
        Create a comprehensive PDF report with all results and visualizations
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"CLT_Experiment_Report_{timestamp}.pdf"
        
        # Ensure we're in the right directory
        filepath = os.path.join("/Users/m0c013f/Documents/next-gen-learning/Statistic", filename)
        
        print(f"\n📄 CREATING PDF REPORT: {filename}")
        print("="*60)
        
        with PdfPages(filepath) as pdf:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Page 1: Title page and experiment info
            self._create_title_page(pdf, results_df)
            
            # Page 2-3: Summary tables for each distribution
            self._create_summary_tables_pdf(pdf, results_df)
            
            # Page 4-5: P-value heatmaps
            self._create_heatmap_page(pdf, results_df)
            
            # Page 6+: Distribution comparisons for each distribution
            self._create_distribution_pages(pdf, results_df)
            
            # Final page: Statistics evolution
            self._create_evolution_page(pdf, results_df)
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = 'Central Limit Theorem Experiment Report'
            d['Author'] = 'CLT Experiment Script'
            d['Subject'] = 'Statistical Analysis of Sample Size Effects on Normality'
            d['Keywords'] = 'Central Limit Theorem, Statistics, Normality Tests'
            d['CreationDate'] = datetime.now()
        
        print(f"✅ PDF Report saved as: {filepath}")
        return filepath
    
    def _plot_pvalue_heatmap(self, results_df):
        """
        Create heatmap of p-values
        """
        # Pivot for heatmap
        shapiro_pivot = results_df.pivot(index='distribution', 
                                       columns='sample_size', 
                                       values='shapiro_pvalue')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Shapiro-Wilk heatmap
        sns.heatmap(shapiro_pivot, annot=True, cmap='RdYlGn', center=0.05,
                   ax=ax1, cbar_kws={'label': 'p-value'})
        ax1.set_title('Shapiro-Wilk Test P-values\n(Green: Normal, Red: Not Normal)')
        ax1.set_xlabel('Sample Size')
        ax1.set_ylabel('Distribution Type')
        
        # KS test heatmap
        ks_pivot = results_df.pivot(index='distribution', 
                                  columns='sample_size', 
                                  values='ks_pvalue')
        
        sns.heatmap(ks_pivot, annot=True, cmap='RdYlGn', center=0.05,
                   ax=ax2, cbar_kws={'label': 'p-value'})
        ax2.set_title('Kolmogorov-Smirnov Test P-values\n(Green: Normal, Red: Not Normal)')
        ax2.set_xlabel('Sample Size')
        ax2.set_ylabel('Distribution Type')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_distribution_comparison(self, results_df):
        """
        Plot actual distributions for key sample sizes
        """
        key_sizes = [5, 30, 100]  # Small, threshold, large
        
        for dist in self.distributions:
            fig, axes = plt.subplots(len(key_sizes), 2, figsize=(15, 4*len(key_sizes)))
            fig.suptitle(f'{dist.upper()} Distribution - Sample Means Distribution', 
                        fontsize=16, y=0.98)
            
            for i, sample_size in enumerate(key_sizes):
                # Get the sample means for this combination
                row_data = results_df[(results_df['distribution'] == dist) & 
                                    (results_df['sample_size'] == sample_size)]
                
                if not row_data.empty:
                    sample_means = row_data.iloc[0]['sample_means']
                    
                    # Histogram
                    sns.histplot(sample_means, kde=True, ax=axes[i, 0], 
                               stat='density', alpha=0.7)
                    axes[i, 0].axvline(np.mean(sample_means), color='red', 
                                     linestyle='--', label=f'Mean: {np.mean(sample_means):.3f}')
                    axes[i, 0].set_title(f'Sample Size: {sample_size}')
                    axes[i, 0].legend()
                    axes[i, 0].grid(True, alpha=0.3)
                    
                    # Q-Q plot
                    stats.probplot(sample_means, dist="norm", plot=axes[i, 1])
                    axes[i, 1].set_title(f'Q-Q Plot (Sample Size: {sample_size})')
                    axes[i, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def _plot_statistics_evolution(self, results_df):
        """
        Plot how statistics evolve with sample size
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Evolution of Sample Mean Statistics with Sample Size', 
                    fontsize=16)
        
        # Plot each statistic
        stats_to_plot = ['skewness', 'kurtosis', 'shapiro_pvalue', 'std']
        titles = ['Skewness Evolution', 'Kurtosis Evolution', 
                 'Shapiro-Wilk p-value Evolution', 'Standard Deviation Evolution']
        
        for i, (stat, title) in enumerate(zip(stats_to_plot, titles)):
            ax = axes[i//2, i%2]
            
            for dist in self.distributions:
                dist_data = results_df[results_df['distribution'] == dist]
                ax.plot(dist_data['sample_size'], dist_data[stat], 
                       marker='o', label=dist, linewidth=2)
            
            ax.set_title(title)
            ax.set_xlabel('Sample Size')
            ax.set_ylabel(stat.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add reference line for p-value
            if stat == 'shapiro_pvalue':
                ax.axhline(y=0.05, color='red', linestyle='--', 
                          label='α = 0.05', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def _create_title_page(self, pdf, results_df):
        """
        Create a title page with experiment overview
        """
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.9, 'Central Limit Theorem\nExperiment Report', 
                ha='center', va='center', fontsize=24, fontweight='bold',
                transform=ax.transAxes)
        
        # Date
        ax.text(0.5, 0.8, f'Generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}', 
                ha='center', va='center', fontsize=12,
                transform=ax.transAxes)
        
        # Experiment parameters
        params_text = f"""
Experiment Parameters:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Population Size: {self.population_size:,}
Number of Samples per Test: {self.num_samples:,}
Sample Sizes Tested: {self.sample_sizes}
Distributions Tested: {self.distributions}

Total Experiments Run: {len(results_df)}

Objective:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This experiment tests the Central Limit Theorem by examining
how sample size affects the normality of sample means across
different population distributions.

Key Questions:
• At what sample size do sample means become normally distributed?
• How does this threshold vary across different distributions?
• How do skewness and kurtosis change with sample size?

Statistical Tests Used:
• Shapiro-Wilk test for normality
• Kolmogorov-Smirnov test for normality
• Descriptive statistics (mean, std, skewness, kurtosis)
        """
        
        ax.text(0.1, 0.6, params_text, ha='left', va='top', fontsize=10,
                transform=ax.transAxes, family='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_summary_tables_pdf(self, pdf, results_df):
        """
        Create summary tables for each distribution
        """
        for dist in self.distributions:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis('off')
            
            # Title
            ax.text(0.5, 0.95, f'{dist.upper()} DISTRIBUTION - RESULTS SUMMARY', 
                    ha='center', va='top', fontsize=16, fontweight='bold',
                    transform=ax.transAxes)
            
            # Get data for this distribution
            dist_data = results_df[results_df['distribution'] == dist].copy()
            
            # Create summary table
            display_df = dist_data[['sample_size', 'mean', 'std', 'skewness', 
                                   'kurtosis', 'shapiro_pvalue', 'ks_pvalue']].copy()
            
            # Add normality verdict
            display_df['normality'] = display_df['shapiro_pvalue'].apply(
                lambda x: 'Normal' if x > 0.05 else 'Not Normal'
            )
            
            # Rename columns
            display_df.columns = ['Sample\nSize', 'Mean', 'Std Dev', 'Skewness', 
                                'Kurtosis', 'Shapiro\np-value', 'KS\np-value', 'Verdict']
            
            # Create table
            table = ax.table(cellText=display_df.round(4).values,
                           colLabels=display_df.columns,
                           cellLoc='center',
                           loc='center',
                           bbox=[0.1, 0.3, 0.8, 0.6])
            
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 2)
            
            # Color code the verdict column
            for i in range(len(display_df)):
                if display_df.iloc[i]['Verdict'] == 'Normal':
                    table[(i+1, 7)].set_facecolor('#90EE90')  # Light green
                else:
                    table[(i+1, 7)].set_facecolor('#FFB6C1')  # Light red
            
            # Add summary text
            normal_count = sum(dist_data['shapiro_pvalue'] > 0.05)
            threshold_sample = dist_data[dist_data['shapiro_pvalue'] > 0.05]['sample_size'].min()
            
            summary_text = f"""
Summary for {dist.upper()} Distribution:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Normal distributions achieved: {normal_count}/{len(dist_data)} sample sizes
• Normality threshold: Sample size ≥ {threshold_sample if pd.notna(threshold_sample) else 'Not achieved'}
• Mean skewness across all sample sizes: {dist_data['skewness'].mean():.4f}
• Mean kurtosis across all sample sizes: {dist_data['kurtosis'].mean():.4f}

Interpretation:
The Central Limit Theorem predicts that sample means should become normally 
distributed as sample size increases, regardless of the population distribution.
Green cells indicate normal distributions (p > 0.05), red cells indicate 
non-normal distributions (p ≤ 0.05).
            """
            
            ax.text(0.1, 0.25, summary_text, ha='left', va='top', fontsize=10,
                    transform=ax.transAxes, family='monospace')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def _create_heatmap_page(self, pdf, results_df):
        """
        Create heatmap visualizations
        """
        # Pivot for heatmap
        shapiro_pivot = results_df.pivot(index='distribution', 
                                       columns='sample_size', 
                                       values='shapiro_pvalue')
        
        ks_pivot = results_df.pivot(index='distribution', 
                                  columns='sample_size', 
                                  values='ks_pvalue')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
        
        # Shapiro-Wilk heatmap
        sns.heatmap(shapiro_pivot, annot=True, cmap='RdYlGn', center=0.05,
                   ax=ax1, cbar_kws={'label': 'p-value'}, fmt='.3f')
        ax1.set_title('Shapiro-Wilk Test P-values\n(Green: Normal ≥ 0.05, Red: Not Normal < 0.05)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sample Size')
        ax1.set_ylabel('Distribution Type')
        
        # KS test heatmap
        sns.heatmap(ks_pivot, annot=True, cmap='RdYlGn', center=0.05,
                   ax=ax2, cbar_kws={'label': 'p-value'}, fmt='.3f')
        ax2.set_title('Kolmogorov-Smirnov Test P-values\n(Green: Normal ≥ 0.05, Red: Not Normal < 0.05)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Sample Size')
        ax2.set_ylabel('Distribution Type')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_distribution_pages(self, pdf, results_df):
        """
        Create distribution comparison pages
        """
        key_sizes = [5, 30, 100]  # Small, threshold, large
        
        for dist in self.distributions:
            fig, axes = plt.subplots(len(key_sizes), 2, figsize=(11, 8.5))
            fig.suptitle(f'{dist.upper()} Distribution - Sample Means Distribution Evolution', 
                        fontsize=16, fontweight='bold')
            
            for i, sample_size in enumerate(key_sizes):
                # Get the sample means for this combination
                row_data = results_df[(results_df['distribution'] == dist) & 
                                    (results_df['sample_size'] == sample_size)]
                
                if not row_data.empty:
                    sample_means = row_data.iloc[0]['sample_means']
                    p_value = row_data.iloc[0]['shapiro_pvalue']
                    
                    # Histogram
                    sns.histplot(sample_means, kde=True, ax=axes[i, 0], 
                               stat='density', alpha=0.7)
                    axes[i, 0].axvline(np.mean(sample_means), color='red', 
                                     linestyle='--', label=f'Mean: {np.mean(sample_means):.3f}')
                    
                    # Add normality status to title
                    status = "✅ Normal" if p_value > 0.05 else "❌ Not Normal"
                    axes[i, 0].set_title(f'Sample Size: {sample_size} | {status} (p={p_value:.3f})')
                    axes[i, 0].legend()
                    axes[i, 0].grid(True, alpha=0.3)
                    
                    # Q-Q plot
                    stats.probplot(sample_means, dist="norm", plot=axes[i, 1])
                    axes[i, 1].set_title(f'Q-Q Plot (Sample Size: {sample_size})')
                    axes[i, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    def _create_evolution_page(self, pdf, results_df):
        """
        Create statistics evolution page
        """
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Evolution of Sample Mean Statistics with Sample Size', 
                    fontsize=16, fontweight='bold')
        
        # Plot each statistic
        stats_to_plot = ['skewness', 'kurtosis', 'shapiro_pvalue', 'std']
        titles = ['Skewness Evolution', 'Kurtosis Evolution', 
                 'Shapiro-Wilk p-value Evolution', 'Standard Deviation Evolution']
        
        for i, (stat, title) in enumerate(zip(stats_to_plot, titles)):
            ax = axes[i//2, i%2]
            
            for dist in self.distributions:
                dist_data = results_df[results_df['distribution'] == dist]
                ax.plot(dist_data['sample_size'], dist_data[stat], 
                       marker='o', label=dist, linewidth=2, markersize=6)
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Sample Size')
            ax.set_ylabel(stat.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add reference line for p-value
            if stat == 'shapiro_pvalue':
                ax.axhline(y=0.05, color='red', linestyle='--', 
                          label='α = 0.05', alpha=0.7)
                ax.legend()
            
            # Add reference lines for skewness and kurtosis
            if stat in ['skewness', 'kurtosis']:
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def run_complete_experiment(self):
        """
        Run the complete experiment with all visualizations and summaries
        """
        print("🚀 STARTING COMPREHENSIVE CLT EXPERIMENT")
        print("This will test the Central Limit Theorem across:")
        print(f"  • {len(self.distributions)} different distributions")
        print(f"  • {len(self.sample_sizes)} different sample sizes")
        print(f"  • Population size: {self.population_size:,}")
        print(f"  • Samples per test: {self.num_samples:,}")
        print("\n" + "="*60)
        
        # Run all experiments
        results_df = self.run_all_experiments()
        
        # Create summary table
        self.create_summary_table(results_df)
        
        # Create visualizations
        print("\n🎨 CREATING VISUALIZATIONS...")
        self.create_visualizations(results_df)
        
        # Create PDF report
        print("\n📄 CREATING PDF REPORT...")
        pdf_file = self.create_pdf_report(results_df)
        
        print("\n✅ EXPERIMENT COMPLETE!")
        print("Check the visualizations above and the comprehensive PDF report.")
        print(f"📄 PDF Report saved at: {pdf_file}")
        
        return results_df


def quick_experiment():
    """
    Run a quick version with fewer parameters for testing
    """
    print("🏃‍♂️ RUNNING QUICK EXPERIMENT")
    
    experiment = CLTExperiment()
    experiment.sample_sizes = [5, 20, 50]  # Fewer sample sizes
    experiment.distributions = ['normal', 'exponential', 'poisson']  # Fewer distributions
    experiment.population_size = 10000  # Smaller population
    experiment.num_samples = 500  # Fewer samples
    
    # Run experiment
    results_df = experiment.run_all_experiments()
    experiment.create_summary_table(results_df)
    
    # Create PDF report for quick experiment
    pdf_file = experiment.create_pdf_report(results_df, "CLT_Quick_Experiment_Report.pdf")
    print(f"\n📄 Quick experiment PDF saved at: {pdf_file}")
    
    return results_df


if __name__ == "__main__":
    # Ask user for experiment type
    choice = input("Choose experiment type:\n"
                  "1. Complete experiment (c) - All distributions and sample sizes\n"
                  "2. Quick experiment (q) - Subset for testing\n"
                  "Enter choice (c/q): ").strip().lower()
    
    if choice in ['q', 'quick']:
        results = quick_experiment()
    else:
        experiment = CLTExperiment()
        results = experiment.run_complete_experiment()
    
    print("\n📊 Results DataFrame shape:", results.shape)
    print("📋 Results saved in 'results' variable for further analysis")