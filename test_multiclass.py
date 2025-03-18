from app import create_app
from app.services.data_processor import prepare_data_for_model, create_combined_dataset
import pandas as pd

def main():
    print("Starting test...")
    app = create_app()
    print("App created")
    with app.app_context():
        print("Testing multiclass changes...")
        
        # Create a combined dataset for 2023
        print("Creating dataset...")
        dataset = create_combined_dataset(2023)
        print(f"Dataset created: {dataset is not None}")
        
        if dataset is not None:
            # Prepare data for model training
            print("Preparing data for model...")
            X, y, features = prepare_data_for_model(dataset, 'indeks_pembangunan_manusia')
            print(f"Data prepared: X shape = {X.shape}, y shape = {len(y)}")
            
            # Check target values
            print(f"Target values distribution: \n{pd.Series(y).value_counts()}")
            print(f"Unique target values: {pd.Series(y).unique()}")
            
            # Check class mapping
            print("\nClass mapping in target variable:")
            sejahtera_value = pd.Series(y)[dataset['label_sejahtera_indeks_pembangunan_manusia'] == 'Sejahtera'].iloc[0]
            menengah_value = pd.Series(y)[dataset['label_sejahtera_indeks_pembangunan_manusia'] == 'Menengah'].iloc[0]
            
            print(f"'Sejahtera' is mapped to: {sejahtera_value}")
            print(f"'Menengah' is mapped to: {menengah_value}")
            
            # We don't have 'Tidak Sejahtera' examples in the dataset
            print("'Tidak Sejahtera' should be mapped to: 0")
        else:
            print("No dataset found for 2023")

if __name__ == "__main__":
    main() 