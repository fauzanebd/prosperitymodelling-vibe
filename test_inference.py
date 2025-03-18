from app import create_app
import numpy as np

def main():
    print("Testing inference with multiclass...")
    
    # Test scenarios:
    test_cases = [
        # Case 1: Binary classifier returns class 2 but only has 2 probability columns
        {
            'y_pred': np.array([2]),
            'y_prob': np.zeros((1, 2)),  # Only 2 probability columns
            'name': 'Binary classifier with class 2'
        },
        # Case 2: Multi-class classifier returns class 2 with 3 probability columns
        {
            'y_pred': np.array([2]),
            'y_prob': np.zeros((1, 3)),  # 3 probability columns
            'name': 'Multi-class classifier with class 2'
        },
        # Case 3: Binary classifier returns class 1
        {
            'y_pred': np.array([1]),
            'y_prob': np.zeros((1, 2)),  # Only 2 probability columns
            'name': 'Binary classifier with class 1'
        },
    ]
    
    for idx, test_case in enumerate(test_cases):
        print(f"\nTest Case {idx+1}: {test_case['name']}")
        y_pred = test_case['y_pred']
        y_prob = test_case['y_prob']
        
        # Original approach (prone to errors)
        try:
            # Try to access probability directly (will fail if out of bounds)
            prob_value = y_prob[0, int(y_pred[0])]
            print(f"  Original approach: Success, probability = {prob_value}")
        except IndexError as e:
            print(f"  Original approach: Failed with error: {e}")
        
        # Robust approach 1: Check dimensions first
        try:
            # Check if y_prob has enough columns for all classes
            if y_prob.shape[1] <= int(y_pred.max()):
                # Not enough columns, create a new array with more columns
                new_y_prob = np.zeros((len(y_pred), int(y_pred.max()) + 1))
                # Copy existing probabilities
                for i in range(min(y_prob.shape[1], new_y_prob.shape[1])):
                    new_y_prob[:, i] = y_prob[:, i]
                # Set the probability for the predicted class to 1.0
                for i, pred in enumerate(y_pred):
                    new_y_prob[i, int(pred)] = 1.0
                y_prob = new_y_prob
            
            # Now access the probability safely
            prob_value = y_prob[0, int(y_pred[0])]
            print(f"  Robust approach 1: Success, probability = {prob_value}")
        except Exception as e:
            print(f"  Robust approach 1: Failed with error: {e}")
        
        # Robust approach 2: Safe indexing with fallback
        try:
            # Use conditional to avoid IndexError
            prediction_prob = y_prob[0, int(y_pred[0])] if int(y_pred[0]) < y_prob.shape[1] else 1.0
            print(f"  Robust approach 2: Success, probability = {prediction_prob}")
        except Exception as e:
            print(f"  Robust approach 2: Failed with error: {e}")

if __name__ == "__main__":
    main()