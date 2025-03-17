import joblib
import pandas as pd
from datasetloader import load_data
import os

def predict(model_path='best_model.pkl', scaler_path='scaler.pkl', output_path='submission.csv'):
    try:
        # Check if model and scaler files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found. Please train the model first.")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file '{scaler_path}' not found. Please train the model first.")
        
        # Load the trained model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load test data
        _, _, X_test, passenger_ids = load_data()
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        predictions = model.predict(X_test_scaled)
        
        # Create submission file
        submission = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Survived': predictions
        })
        
        # Save predictions
        submission.to_csv(output_path, index=False)
        print(f"Predictions saved to '{output_path}'")
        
        # Print prediction statistics
        total_passengers = len(predictions)
        survivors = sum(predictions)
        print(f"\nPrediction Statistics:")
        print(f"Total passengers: {total_passengers}")
        print(f"Predicted survivors: {survivors}")
        print(f"Predicted fatalities: {total_passengers - survivors}")
        print(f"Survival rate: {(survivors/total_passengers)*100:.2f}%")
        
        return predictions, submission
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None

if __name__ == "__main__":
    predictions, submission = predict()
    if predictions is not None:
        print("\nFirst 5 predictions:")
        print(submission.head())