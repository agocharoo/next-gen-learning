import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def data_generator(distribution_type='normal', data_points=1000):
    """
    Step 1: Data Generator
    
    Generates random data points from specified distribution
    
    Parameters:
    distribution_type (str): Type of distribution ('normal', 'bernoulli', 'poisson', 'exponential', 'beta')
    data_points (int): Number of data points to generate (default: 1000)
    
    Returns:
    list: Generated data points
    """
    np.random.seed(42)  # For reproducibility
    
    distribution_type = distribution_type.lower()
    
    if distribution_type == 'normal':
        # Normal distribution with mean=0, std=1
        data = np.random.normal(loc=0, scale=1, size=data_points)
    
    elif distribution_type == 'bernoulli':
        # Bernoulli distribution with p=0.5
        data = np.random.binomial(n=1, p=0.5, size=data_points)
    
    elif distribution_type == 'poisson':
        # Poisson distribution with lambda=3
        data = np.random.poisson(lam=3, size=data_points)
    
    elif distribution_type == 'exponential':
        # Exponential distribution with scale=1
        data = np.random.exponential(scale=1, size=data_points)
    
    elif distribution_type == 'beta':
        # Beta distribution with alpha=2, beta=5
        data = np.random.beta(a=2, b=5, size=data_points)
    
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")
    
    print(f"Generated {data_points} data points from {distribution_type} distribution")
    print(f"Population mean: {np.mean(data):.4f}")
    print(f"Population std: {np.std(data):.4f}")
    
    return data.tolist()


def data_collection(sample_size, num_samples, population_data):
    """
    Step 2: Data Collection (Sampling)
    
    Randomly samples from population data and calculates sample means
    
    Parameters:
    sample_size (int): Size of each sample
    num_samples (int): Number of random samples to take
    population_data (list): The population data to sample from
    
    Returns:
    list: List of sample means
    """
    if sample_size > len(population_data):
        raise ValueError("Sample size cannot be larger than population size")
    
    sample_means = []
    
    for i in range(num_samples):
        # Randomly sample without replacement
        sample = np.random.choice(population_data, size=sample_size, replace=False)
        sample_mean = np.mean(sample)
        sample_means.append(sample_mean)
    
    print(f"\nSampling completed:")
    print(f"Number of samples: {num_samples}")
    print(f"Sample size: {sample_size}")
    print(f"Mean of sample means: {np.mean(sample_means):.4f}")
    print(f"Standard deviation of sample means: {np.std(sample_means):.4f}")
    print(f"Standard error (theoretical): {np.std(population_data)/np.sqrt(sample_size):.4f}")
    
    return sample_means


def plot_distributions(sample_means, title_suffix=""):
    """
    Step 3: Plotting
    
    Creates distribution plot and QQ plot to check normality
    
    Parameters:
    sample_means (list): List of sample means to plot
    title_suffix (str): Additional text for plot title
    """
    # Set up the figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Distribution of sample means
    sns.histplot(sample_means, kde=True, ax=ax1, stat='density', alpha=0.7)
    ax1.axvline(np.mean(sample_means), color='red', linestyle='--', 
                label=f'Mean: {np.mean(sample_means):.4f}')
    ax1.set_title(f'Distribution of Sample Means{title_suffix}')
    ax1.set_xlabel('Sample Mean')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: QQ plot for normality check
    stats.probplot(sample_means, dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot (Normality Check){title_suffix}')
    ax2.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('Central Limit Theorem Demonstration', fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.show()
    
    # Perform normality tests with enhanced explanation
    enhanced_normality_explanation(sample_means, title_suffix.replace(" - ", "").replace(" Population", ""))


def demonstrate_central_limit_theorem():
    """
    Complete demonstration of Central Limit Theorem
    """
    print("=" * 60)
    print("CENTRAL LIMIT THEOREM DEMONSTRATION")
    print("=" * 60)
    
    # Step 1: Generate population data
    print("\nSTEP 1: DATA GENERATION")
    print("-" * 30)
    
    # Let's try with different distributions
    distributions = ['exponential', 'poisson', 'beta']
    
    for dist in distributions:
        print(f"\n--- Testing with {dist.upper()} distribution ---")
        
        # Generate population data
        population = data_generator(distribution_type=dist, data_points=10000)
        
        # Step 2: Sample collection
        print("\nSTEP 2: DATA COLLECTION (SAMPLING)")
        print("-" * 35)
        sample_means = data_collection(sample_size=30, num_samples=1000, 
                                     population_data=population)
        
        # Step 3: Plotting
        print("\nSTEP 3: VISUALIZATION")
        print("-" * 20)
        plot_distributions(sample_means, f" - {dist.capitalize()} Population")
        
        print("\n" + "="*60)


def interactive_demo():
    """
    Interactive version for user input
    """
    print("=" * 60)
    print("INTERACTIVE CENTRAL LIMIT THEOREM DEMO")
    print("=" * 60)
    
    # Get user input
    print("\nAvailable distributions: normal, bernoulli, poisson, exponential, beta")
    dist_type = input("Enter distribution type (default: normal): ").strip() or 'normal'
    
    try:
        data_points = int(input("Enter number of data points (default: 5000): ") or 5000)
        sample_size = int(input("Enter sample size (default: 30): ") or 30)
        num_samples = int(input("Enter number of samples (default: 500): ") or 500)
    except ValueError:
        print("Using default values...")
        data_points, sample_size, num_samples = 5000, 30, 500
    
    # Execute the three steps
    print(f"\nGenerating {data_points} points from {dist_type} distribution...")
    population = data_generator(distribution_type=dist_type, data_points=data_points)
    
    print(f"\nTaking {num_samples} samples of size {sample_size}...")
    sample_means = data_collection(sample_size=sample_size, num_samples=num_samples, 
                                 population_data=population)
    
    print("\nCreating visualizations...")
    plot_distributions(sample_means, f" - {dist_type.capitalize()} Population")


def test_sample_size_effect():
    """
    Demonstrates how sample size affects normality of sample means
    """
    print("=" * 60)
    print("SAMPLE SIZE EFFECT ON NORMALITY")
    print("=" * 60)
    
    # Generate a highly non-normal population (exponential)
    population = data_generator(distribution_type='exponential', data_points=10000)
    
    # Test different sample sizes
    sample_sizes = [5, 10, 20, 30, 50, 100]
    
    for sample_size in sample_sizes:
        print(f"\n--- Testing with sample size: {sample_size} ---")
        
        # Collect sample means
        sample_means = data_collection(sample_size=sample_size, num_samples=500, 
                                     population_data=population)
        
        # Quick normality test
        shapiro_stat, shapiro_p = stats.shapiro(sample_means)
        
        print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}", end=" ")
        if shapiro_p > 0.05:
            print("✅ NORMAL")
        else:
            print("❌ NOT NORMAL")
        
        # Create plots for interesting cases
        if sample_size in [5, 30, 100]:
            plot_distributions(sample_means, f" - Sample Size {sample_size}")


def enhanced_normality_explanation(sample_means, distribution_name=""):
    """
    Enhanced explanation of normality test results
    """
    shapiro_stat, shapiro_p = stats.shapiro(sample_means)
    ks_stat, ks_p = stats.kstest(sample_means, 'norm', 
                                args=(np.mean(sample_means), np.std(sample_means)))
    
    print(f"\n" + "="*50)
    print("NORMALITY TEST RESULTS EXPLANATION")
    print("="*50)
    
    print(f"\n📊 Distribution tested: {distribution_name}")
    print(f"📈 Number of sample means: {len(sample_means)}")
    
    print(f"\n🔍 SHAPIRO-WILK TEST:")
    print(f"   Statistic: {shapiro_stat:.4f}")
    print(f"   P-value: {shapiro_p:.4f}")
    
    print(f"\n🔍 KOLMOGOROV-SMIRNOV TEST:")
    print(f"   Statistic: {ks_stat:.4f}")
    print(f"   P-value: {ks_p:.4f}")
    
    print(f"\n📖 INTERPRETATION:")
    print(f"   • Null hypothesis (H₀): Data is normally distributed")
    print(f"   • Alternative hypothesis (H₁): Data is NOT normally distributed")
    print(f"   • Significance level (α): 0.05")
    
    if shapiro_p > 0.05:
        print(f"\n✅ RESULT: NORMAL DISTRIBUTION")
        print(f"   • P-value ({shapiro_p:.4f}) > α (0.05)")
        print(f"   • We FAIL TO REJECT H₀")
        print(f"   • Sample means appear to be normally distributed")
        print(f"   • This supports the Central Limit Theorem! 🎉")
    else:
        print(f"\n❌ RESULT: NOT NORMAL DISTRIBUTION")
        print(f"   • P-value ({shapiro_p:.4f}) ≤ α (0.05)")
        print(f"   • We REJECT H₀")
        print(f"   • Sample means do NOT appear normally distributed")
        print(f"   • Try increasing sample size or number of samples")
    
    # Additional insights
    mean_val = np.mean(sample_means)
    std_val = np.std(sample_means)
    skewness = stats.skew(sample_means)
    kurtosis = stats.kurtosis(sample_means)
    
    print(f"\n📊 DESCRIPTIVE STATISTICS:")
    print(f"   • Mean: {mean_val:.4f}")
    print(f"   • Standard deviation: {std_val:.4f}")
    print(f"   • Skewness: {skewness:.4f} ({'symmetric' if abs(skewness) < 0.5 else 'skewed'})")
    print(f"   • Kurtosis: {kurtosis:.4f} ({'normal' if abs(kurtosis) < 0.5 else 'heavy/light tails'})")


if __name__ == "__main__":
    # Set style for better plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Run demonstration
    choice = input("Choose mode:\n1. Full demonstration (d)\n2. Interactive mode (i)\n3. Sample size effect test (s)\nEnter choice (d/i/s): ").strip().lower()
    
    if choice in ['i', 'interactive']:
        interactive_demo()
    elif choice in ['s', 'sample']:
        test_sample_size_effect()
    else:
        demonstrate_central_limit_theorem()