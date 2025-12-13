"""
Statistical Significance Tests
McNemar's Test and Wilcoxon Signed-Rank Test
"""

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
import pandas as pd

def mcnemar_test(predictions_model1, predictions_model2, ground_truth):
    """
    McNemar's Test for comparing two classifiers
    
    Tests null hypothesis: both models have same error rate
    
    Args:
        predictions_model1: Predictions from model 1
        predictions_model2: Predictions from model 2
        ground_truth: True labels
    
    Returns:
        Dictionary with test statistic and p-value
    """
    print("Performing McNemar's Test...")
    
    predictions_model1 = np.array(predictions_model1)
    predictions_model2 = np.array(predictions_model2)
    ground_truth = np.array(ground_truth)
    
    # Calculate correctness
    correct_1 = (predictions_model1 == ground_truth)
    correct_2 = (predictions_model2 == ground_truth)
    
    # Create contingency table
    both_correct = np.sum(correct_1 & correct_2)
    both_wrong = np.sum(~correct_1 & ~correct_2)
    model1_correct_model2_wrong = np.sum(correct_1 & ~correct_2)
    model1_wrong_model2_correct = np.sum(~correct_1 & correct_2)
    
    # Contingency table format:
    # [[both_correct, model1_correct_model2_wrong],
    #  [model1_wrong_model2_correct, both_wrong]]
    
    table = [[both_correct, model1_correct_model2_wrong],
             [model1_wrong_model2_correct, both_wrong]]
    
    print(f"\nContingency Table:")
    print(f"  Both correct:          {both_correct}")
    print(f"  Both wrong:            {both_wrong}")
    print(f"  Model 1 right, 2 wrong: {model1_correct_model2_wrong}")
    print(f"  Model 1 wrong, 2 right: {model1_wrong_model2_correct}")
    
    # Perform McNemar's test
    result = mcnemar(table, exact=False, correction=True)
    
    print(f"\nMcNemar's Test Results:")
    print(f"  Test statistic: {result.statistic:.4f}")
    print(f"  P-value:        {result.pvalue:.4f}")
    
    if result.pvalue < 0.05:
        print(f"  ✓ Significant difference (p < 0.05)")
    else:
        print(f"  ✗ No significant difference (p >= 0.05)")
    
    return {
        'statistic': result.statistic,
        'pvalue': result.pvalue,
        'significant': result.pvalue < 0.05,
        'contingency_table': table
    }

def wilcoxon_test(scores_model1, scores_model2, alternative='two-sided'):
    """
    Wilcoxon Signed-Rank Test for paired samples
    
    Non-parametric test for comparing two related samples
    
    Args:
        scores_model1: Performance scores from model 1
        scores_model2: Performance scores from model 2
        alternative: 'two-sided', 'less', or 'greater'
    
    Returns:
        Dictionary with test statistic and p-value
    """
    print("\nPerforming Wilcoxon Signed-Rank Test...")
    
    scores_model1 = np.array(scores_model1)
    scores_model2 = np.array(scores_model2)
    
    # Calculate differences
    differences = scores_model2 - scores_model1
    
    print(f"\nScore Statistics:")
    print(f"  Model 1 mean: {np.mean(scores_model1):.4f} ± {np.std(scores_model1):.4f}")
    print(f"  Model 2 mean: {np.mean(scores_model2):.4f} ± {np.std(scores_model2):.4f}")
    print(f"  Mean difference: {np.mean(differences):.4f}")
    
    # Perform Wilcoxon test
    try:
        statistic, pvalue = wilcoxon(scores_model1, scores_model2, 
                                     alternative=alternative, zero_method='zsplit')
        
        print(f"\nWilcoxon Signed-Rank Test Results:")
        print(f"  Test statistic: {statistic:.4f}")
        print(f"  P-value:        {pvalue:.4f}")
        
        if pvalue < 0.01:
            print(f"  ✓ Highly significant difference (p < 0.01)")
        elif pvalue < 0.05:
            print(f"  ✓ Significant difference (p < 0.05)")
        else:
            print(f"  ✗ No significant difference (p >= 0.05)")
        
        return {
            'statistic': statistic,
            'pvalue': pvalue,
            'significant_0.05': pvalue < 0.05,
            'significant_0.01': pvalue < 0.01,
            'mean_diff': np.mean(differences)
        }
    except ValueError as e:
        print(f"⚠️  Cannot perform test: {e}")
        return None

def compare_models_statistical(model1_results, model2_results, model1_name="Model 1", 
                               model2_name="Model 2"):
    """
    Comprehensive statistical comparison of two models
    
    Args:
        model1_results: Dict with 'predictions', 'ground_truth', 'scores' (optional)
        model2_results: Dict with 'predictions', 'ground_truth', 'scores' (optional)
        model1_name: Name of model 1
        model2_name: Name of model 2
    
    Returns:
        Dictionary with all test results
    """
    print("="*70)
    print(f" STATISTICAL COMPARISON: {model1_name} vs {model2_name}")
    print("="*70)
    
    results = {}
    
    # McNemar's Test (for predictions)
    if 'predictions' in model1_results and 'predictions' in model2_results:
        ground_truth = model1_results['ground_truth']
        
        mcnemar_results = mcnemar_test(
            model1_results['predictions'],
            model2_results['predictions'],
            ground_truth
        )
        results['mcnemar'] = mcnemar_results
    
    # Wilcoxon Test (for continuous scores, e.g., per-sample F1 or accuracy)
    if 'scores' in model1_results and 'scores' in model2_results:
        wilcoxon_results = wilcoxon_test(
            model1_results['scores'],
            model2_results['scores']
        )
        results['wilcoxon'] = wilcoxon_results
    
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    if 'mcnemar' in results:
        print(f"McNemar's Test:  p = {results['mcnemar']['pvalue']:.4f} ", end="")
        print("(SIGNIFICANT)" if results['mcnemar']['significant'] else "(not significant)")
    
    if 'wilcoxon' in results and results['wilcoxon']:
        print(f"Wilcoxon Test:   p = {results['wilcoxon']['pvalue']:.4f} ", end="")
        print("(SIGNIFICANT)" if results['wilcoxon']['significant_0.05'] else "(not significant)")
    
    print("="*70 + "\n")
    
    return results

def bootstrap_confidence_interval(scores, n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence interval for metric
    
    Args:
        scores: Array of scores
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95)
    
    Returns:
        (lower_bound, upper_bound, mean)
    """
    scores = np.array(scores)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    mean = np.mean(scores)
    
    return lower, upper, mean

def create_comparison_table(results_dict, save_path='results/tables/model_comparison.csv'):
    """
    Create comparison table for multiple models
    
    Args:
        results_dict: Dictionary of {model_name: {metrics}}
        save_path: Where to save CSV
    """
    from pathlib import Path
    
    # Extract metrics
    data = []
    for model_name, metrics in results_dict.items():
        row = {'Model': model_name}
        row.update(metrics)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Reorder columns
    col_order = ['Model', 'accuracy', 'precision', 'recall', 'f1_score', 
                 'sensitivity', 'specificity']
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order + [c for c in df.columns if c not in col_order]]
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False, float_format='%.4f')
    
    print(f"✓ Comparison table saved to {save_path}")
    print("\n" + df.to_string(index=False))
    
    return df

if __name__ == '__main__':
    # Example usage
    print("Statistical Tests Module")
    print("="*70)
    print("\nExample: Comparing baseline vs GAN-augmented model\n")
    
    # Simulated data for demonstration
    np.random.seed(42)
    n_samples = 100
    
    # Baseline model (lower performance)
    baseline_preds = np.random.binomial(1, 0.7, n_samples)
    ground_truth = np.random.binomial(1, 0.5, n_samples)
    
    # GAN-augmented model (better performance)
    gan_preds = np.where(
        np.random.random(n_samples) < 0.85,
        ground_truth,
        1 - ground_truth
    )
    
    baseline_results = {
        'predictions': baseline_preds,
        'ground_truth': ground_truth,
        'scores': np.random.uniform(0.6, 0.8, n_samples)
    }
    
    gan_results = {
        'predictions': gan_preds,
        'ground_truth': ground_truth,
        'scores': np.random.uniform(0.7, 0.9, n_samples)
    }
    
    # Compare
    results = compare_models_statistical(
        baseline_results,
        gan_results,
        "Baseline",
        "GAN-Augmented"
    )
    
    # Save results
    import json
    from pathlib import Path
    
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    with open('results/tables/statistical_tests.json', 'w') as f:
        # Convert numpy values to Python types
        results_json = {}
        for key, val in results.items():
            if isinstance(val, dict):
                results_json[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                    for k, v in val.items()}
            else:
                results_json[key] = val
        json.dump(results_json, f, indent=4)
    
    print("✓ Results saved to results/tables/statistical_tests.json")
