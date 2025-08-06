import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import itertools

def calculate_metrics(y_true, y_pred_proba_male, y_pred_proba_female, male_th, female_th):
    """
    Calculate metrics for given thresholds
    
    Args:
        y_true: True labels (0 for female, 1 for male)
        y_pred_proba_male: Prediction probabilities for male class
        y_pred_proba_female: Prediction probabilities for female class (1 - male_proba)
        male_th: Threshold for predicting male
        female_th: Threshold for predicting female
    
    Returns:
        Dictionary with all relevant metrics
    """
    # Create predictions based on thresholds
    y_pred = np.full(len(y_pred_proba_male), -1)  # -1 for unknown
    y_pred[y_pred_proba_male >= male_th] = 1      # Male
    y_pred[y_pred_proba_female >= female_th] = 0  # Female
    
    # Calculate rejection rates
    unknown_mask = (y_pred == -1)
    
    # Separate by true gender for rejection rate calculation
    true_male_mask = (y_true == 1)
    true_female_mask = (y_true == 0)
    
    male_rejected = np.sum(unknown_mask & true_male_mask)
    female_rejected = np.sum(unknown_mask & true_female_mask)
    
    total_males = np.sum(true_male_mask)
    total_females = np.sum(true_female_mask)
    
    male_rejection_rate = male_rejected / total_males if total_males > 0 else 1.0
    female_rejection_rate = female_rejected / total_females if total_females > 0 else 1.0
    
    # Filter out unknown predictions for metric calculation
    known_mask = (y_pred != -1)
    if np.sum(known_mask) == 0:
        return {
            'male_rejection_rate': male_rejection_rate,
            'female_rejection_rate': female_rejection_rate,
            'male_f1': 0,
            'female_f1': 0,
            'male_recall': 0,
            'female_recall': 0,
            'male_precision': 0,
            'female_precision': 0,
            'valid': False
        }
    
    y_true_known = y_true[known_mask]
    y_pred_known = y_pred[known_mask]
    
    # Calculate precision, recall, F1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_known, y_pred_known, labels=[0, 1], zero_division=0
    )
    
    return {
        'male_rejection_rate': male_rejection_rate,
        'female_rejection_rate': female_rejection_rate,
        'male_f1': f1[1],
        'female_f1': f1[0],
        'male_recall': recall[1],
        'female_recall': recall[0],
        'male_precision': precision[1],
        'female_precision': precision[0],
        'valid': True
    }

def check_constraints(metrics, max_rejection_rate, precision_min=0.98, recall_min=0.96):
    """
    Check if the metrics satisfy all constraints
    
    Args:
        metrics: Dictionary with calculated metrics
        max_rejection_rate: Maximum allowed rejection rate
        precision_min: Minimum required precision for both genders
        recall_min: Minimum required recall for both genders
    
    Returns:
        Boolean indicating if constraints are satisfied
    """
    if not metrics['valid']:
        return False
    
    # Constraint 2: Both rejection rates should be less than max_rejection_rate
    if (metrics['male_rejection_rate'] > max_rejection_rate or 
        metrics['female_rejection_rate'] > max_rejection_rate):
        return False
    
    # Constraint 3: Female rejection rate should be <= male rejection rate
    if metrics['female_rejection_rate'] > metrics['male_rejection_rate']:
        return False

    # Constraint 4: Both male and female precision should be >= precision_min
    if (metrics['male_precision'] < precision_min or 
        metrics['female_precision'] < precision_min):
        return False
    
    # Constraint 5: Both male and female recall should be >= recall_min
    if (metrics['male_recall'] < recall_min or 
        metrics['female_recall'] < recall_min):
        return False
    
    return True

def find_optimal_thresholds(df, threshold_search_range, max_rejection_rate, precision_min, recall_min, step_size):
    """
    Find optimal thresholds that satisfy constraints and maximize F1 scores
    
    Args:
        df: DataFrame with 'gender' and 'c1' columns
        threshold_search_range: Tuple (min, max) for both male and female threshold range
        max_rejection_rate: Maximum allowed rejection rate
        precision_min: Minimum required precision for both genders
        recall_min: Minimum required recall for both genders
        step_size: Step size for threshold search
    
    Returns:
        Dictionary with optimal thresholds and their metrics
    """
    # Use gender directly as it's already 0 (female) and 1 (male)
    y_true = df['gender'].astype(int)
    y_pred_proba_male = df['c1'].values
    y_pred_proba_female = 1 - y_pred_proba_male  # Convert to female confidence scores
    
    # Generate threshold candidates
    thresholds = np.arange(threshold_search_range[0], threshold_search_range[1] + step_size, step_size)
    
    best_score = -1
    best_thresholds = None
    best_metrics = None
    valid_combinations = []
    
    print(f"Searching through {len(thresholds)} × {len(thresholds)} = {len(thresholds) * len(thresholds)} combinations...")
    
    for male_th, female_th in itertools.product(thresholds, thresholds):
        metrics = calculate_metrics(y_true, y_pred_proba_male, y_pred_proba_female, male_th, female_th)
        
        if check_constraints(metrics, max_rejection_rate, precision_min, recall_min):
            # Calculate combined score (average F1 with preference for female recall)
            combined_f1 = (metrics['male_f1'] + metrics['female_f1']) / 2
            # Add bonus for higher female recall
            score = combined_f1 + 0.1 * metrics['female_recall']
            
            valid_combinations.append({
                'male_th': round(male_th, 3),
                'female_th': round(female_th, 3),
                'score': score,
                'metrics': metrics
            })
            
            if score > best_score:
                best_score = score
                best_thresholds = (round(male_th, 3), round(female_th, 3))
                best_metrics = metrics
    
    print(f"Found {len(valid_combinations)} valid threshold combinations")
    
    if best_thresholds is None:
        print("No valid threshold combination found that satisfies all constraints!")
        return None
    
    return {
        'male_threshold': best_thresholds[0],
        'female_threshold': best_thresholds[1],
        'metrics': best_metrics,
        'score': best_score,
        'all_valid_combinations': valid_combinations
    }

def print_results(result):
    """Print the optimization results in a readable format"""
    if result is None:
        print("No optimal solution found!")
        return
    
    print("\n" + "="*50)
    print("OPTIMAL THRESHOLDS FOUND")
    print("="*50)
    print(f"Male Threshold: {result['male_threshold']:.3f}")
    print(f"Female Threshold: {result['female_threshold']:.3f}")
    print(f"Combined Score: {result['score']:.4f}")
    
    print("\n" + "-"*30)
    print("DETAILED METRICS")
    print("-"*30)
    metrics = result['metrics']
    print(f"Male Rejection Rate: {metrics['male_rejection_rate']:.4f}")
    print(f"Female Rejection Rate: {metrics['female_rejection_rate']:.4f}")
    print(f"Male F1 Score: {metrics['male_f1']:.4f}")
    print(f"Female F1 Score: {metrics['female_f1']:.4f}")
    print(f"Male Recall: {metrics['male_recall']:.4f}")
    print(f"Female Recall: {metrics['female_recall']:.4f}")
    print(f"Male Precision: {metrics['male_precision']:.4f}")
    print(f"Female Precision: {metrics['female_precision']:.4f}")
    
    print(f"\nRecall Difference (Female - Male): {metrics['female_recall'] - metrics['male_recall']:.4f}")
    print(f"Average F1 Score: {(metrics['male_f1'] + metrics['female_f1']) / 2:.4f}")
    print(f"Minimum Precision: {min(metrics['male_precision'], metrics['female_precision']):.4f}")
    print(f"Minimum Recall: {min(metrics['male_recall'], metrics['female_recall']):.4f}")

def debug_constraints(df, max_rejection_rate, precision_min, recall_min):
    """Debug function to understand why constraints are failing"""
    # Use gender directly as it's already 0 (female) and 1 (male)
    y_true = df['gender'].astype(int)
    y_pred_proba_male = df['c1'].values
    y_pred_proba_female = 1 - y_pred_proba_male  # Convert to female confidence scores
    
    print(f"Gender distribution: Male (1)={np.sum(y_true)}, Female (0)={len(y_true)-np.sum(y_true)}")
    print(f"Male percentage: {np.sum(y_true)/len(y_true)*100:.1f}%")
    print(f"Male confidence (c1) stats: min={y_pred_proba_male.min():.3f}, max={y_pred_proba_male.max():.3f}, mean={y_pred_proba_male.mean():.3f}")
    print(f"Female confidence (1-c1) stats: min={y_pred_proba_female.min():.3f}, max={y_pred_proba_female.max():.3f}, mean={y_pred_proba_female.mean():.3f}")
    
    # Show distribution of confidence scores by true gender
    male_scores_male = y_pred_proba_male[y_true == 1]
    female_scores_male = y_pred_proba_male[y_true == 0]
    male_scores_female = y_pred_proba_female[y_true == 1]
    female_scores_female = y_pred_proba_female[y_true == 0]
    
    print(f"True males - Male confidence: mean={male_scores_male.mean():.3f}, std={male_scores_male.std():.3f}")
    print(f"True females - Male confidence: mean={female_scores_male.mean():.3f}, std={female_scores_male.std():.3f}")
    print(f"True males - Female confidence: mean={male_scores_female.mean():.3f}, std={male_scores_female.std():.3f}")
    print(f"True females - Female confidence: mean={female_scores_female.mean():.3f}, std={female_scores_female.std():.3f}")
    
    # Test a few threshold combinations
    test_combinations = [
        (0.75, 0.75),  # Now both thresholds are in terms of confidence for respective classes
        (0.80, 0.80),
    ]
    
    print(f"\nTesting constraint feasibility:")
    print(f"Max rejection rate allowed: {max_rejection_rate}")
    print(f"Minimum precision required: {precision_min}")
    print(f"Minimum recall required: {recall_min}")
    
    for male_th, female_th in test_combinations:
        metrics = calculate_metrics(y_true, y_pred_proba_male, y_pred_proba_female, male_th, female_th)
        print(f"\nThresholds: Male={male_th}, Female={female_th}")
        print(f"  Male rejection rate: {metrics['male_rejection_rate']:.4f}")
        print(f"  Female rejection rate: {metrics['female_rejection_rate']:.4f}")
        print(f"  Male recall: {metrics['male_recall']:.4f}")
        print(f"  Female recall: {metrics['female_recall']:.4f}")
        print(f"  Male precision: {metrics['male_precision']:.4f}")
        print(f"  Female precision: {metrics['female_precision']:.4f}")
        print(f"  Male F1: {metrics['male_f1']:.4f}")
        print(f"  Female F1: {metrics['female_f1']:.4f}")
        print(f"  Constraints satisfied: {check_constraints(metrics, max_rejection_rate, precision_min, recall_min)}")
        
        # Check each constraint individually
        print("  Individual constraint checks:")
        print(f"    Valid metrics: {metrics['valid']}")
        print(f"    Male rejection <= {max_rejection_rate}: {metrics['male_rejection_rate'] <= max_rejection_rate}")
        print(f"    Female rejection <= {max_rejection_rate}: {metrics['female_rejection_rate'] <= max_rejection_rate}")
        print(f"    Female rejection <= Male rejection: {metrics['female_rejection_rate'] <= metrics['male_rejection_rate']}")
        print(f"    Male precision >= {precision_min}: {metrics['male_precision'] >= precision_min}")
        print(f"    Female precision >= {precision_min}: {metrics['female_precision'] >= precision_min}")
        print(f"    Male recall >= {recall_min}: {metrics['male_recall'] >= recall_min}")
        print(f"    Female recall >= {recall_min}: {metrics['female_recall'] >= recall_min}")


#########################
#
def main():
    data_file = 'data/PeerJ_7000.pred.csv'
    #threshold_search_range, step_size = (0.50, 0.65), 0.001  # combined with P/R = 0.9/0.9 and rejection rate=0.035 to get 0.603/0.543
    #threshold_search_range, step_size = (0.60, 0.90), 0.01  # combined with P/R = 0.97/0.95 and rejection rate=0.125 to get 0.82/0.78

    parser = argparse.ArgumentParser(description='Process with configurable rejection rate and performance thresholds')
    
    parser.add_argument('--max-rejection-rate', type=float, default=0.125,
                       help='Maximum rejection rate (default: 0.125)')
    parser.add_argument('--threshold-range-left', type=float, default=0.50,
                       help='Left boundary of threshold search range (default: 0.50)')
    parser.add_argument('--threshold-range-right', type=float, default=0.90,
                       help='Right boundary of threshold search range (default: 0.90)')
    parser.add_argument('--step-size', type=float, default=0.01,
                       help='Step size for threshold search (default: 0.01)')
    parser.add_argument('--precision-min', type=float, default=0.90,
                       help='Minimum precision required for both genders (default: 0.90)')
    parser.add_argument('--recall-min', type=float, default=0.90,
                       help='Minimum recall required for both genders (default: 0.90)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Use the parsed arguments
    max_rejection_rate = args.max_rejection_rate
    precision_min = args.precision_min
    recall_min = args.recall_min
    threshold_search_range = (args.threshold_range_left, args.threshold_range_right)
    step_size = args.step_size
    
    print(f"Max rejection rate: {max_rejection_rate}")
    print(f"Minimum precision: {precision_min}")
    print(f"Minimum recall: {recall_min}")

    print(f"Loading data from {data_file}...")
    try:
        df = pd.read_csv(data_file, usecols=['gender', 'c1'])
        print(f"Data loaded successfully: {len(df)} samples")
        print(f"Gender distribution:")
        print(df['gender'].value_counts())
        print(f"Male prediction score (c1) range: {df['c1'].min():.3f} - {df['c1'].max():.3f}")
        print(f"Female prediction score (1-c1) range: {(1-df['c1']).min():.3f} - {(1-df['c1']).max():.3f}")
    except FileNotFoundError:
        print(f"Error: Could not find file {data_file}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Debug constraints first
    debug = False
    if debug:
        print(f"\n" + "="*50)
        print("DEBUGGING CONSTRAINTS")
        print("="*50)
        debug_constraints(df, max_rejection_rate, precision_min, recall_min)

    # Find optimal thresholds
    print(f"\n" + "="*50)
    print("OPTIMIZING THRESHOLDS")
    print("="*50)
    print(f"Max rejection rate: {max_rejection_rate}")
    print(f"Male threshold range: {threshold_search_range}")
    print(f"Female threshold range: {threshold_search_range}")
    print(f"Minimum precision required: {precision_min}")
    print(f"Minimum recall required: {recall_min}")
    print("Note: Both thresholds now represent confidence for their respective classes")

    result = find_optimal_thresholds(df, threshold_search_range, max_rejection_rate, 
                                   precision_min, recall_min, step_size)

    # Print results
    print_results(result)
    
    # Show top alternatives
    if result and len(result['all_valid_combinations']) > 1:
        print(f"\n" + "-"*30)
        print("TOP 5 ALTERNATIVE SOLUTIONS")
        print("-"*30)
        sorted_combinations = sorted(result['all_valid_combinations'], 
                                   key=lambda x: x['score'], reverse=True)
        for i, combo in enumerate(sorted_combinations[:5]):
            min_precision = min(combo['metrics']['male_precision'], combo['metrics']['female_precision'])
            min_recall = min(combo['metrics']['male_recall'], combo['metrics']['female_recall'])
            print(f"{i+1}. Male_TH: {combo['male_th']:.3f}, Female_TH: {combo['female_th']:.3f}, "
                  f"Score: {combo['score']:.4f}, "
                  f"F1_avg: {(combo['metrics']['male_f1'] + combo['metrics']['female_f1'])/2:.4f}, "
                  f"Min_Prec: {min_precision:.4f}, Min_Recall: {min_recall:.4f}")
    
    # If no solution found, suggest relaxing constraints
    if result is None:
        print(f"\n" + "!"*50)
        print("SUGGESTIONS FOR CONSTRAINT RELAXATION")
        print("!"*50)
        print("Consider:")
        print(f"1. Increasing max_rejection_rate from {max_rejection_rate} to 0.20 or higher")
        print("2. Expanding threshold ranges (e.g., 0.60-0.90)")
        print(f"3. Reducing minimum precision from {precision_min} to 0.95 or 0.90")
        print(f"4. Reducing minimum recall from {recall_min} to 0.90 or 0.85")
        print("5. Check if your data format matches expectations (M/F vs Male/Female vs 1/0)")

if __name__ == "__main__":
    main()