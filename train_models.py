from app import create_app
from app.services.model_trainer import retrain_model_if_needed, generate_predictions
from app.models.ml_models import TrainedModel

def main():
    app = create_app()
    with app.app_context():
        print("Training models...")
        success = retrain_model_if_needed('indeks_pembangunan_manusia')
        
        if success:
            print("Model berhasil dilatih!")
            
            # Get the latest model
            latest_model = TrainedModel.query.order_by(TrainedModel.created_at.desc()).first()
            
            if latest_model:
                # Generate predictions using the latest model
                predictions = generate_predictions(latest_model.id)
                print(f"Generated {len(predictions)} predictions for model {latest_model.model_type}")
            else:
                print("No model found for predictions")
        else:
            print("Failed to train models")

if __name__ == "__main__":
    main() 