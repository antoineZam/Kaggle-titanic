import joblib
import pandas as pd
from datasetloader import load_data

def predict():
    # Load the trained model and scaler
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
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
    submission.to_csv('submission.csv', index=False)
    print("Predictions saved to 'submission.csv'")
    
    return predictions

if __name__ == "__main__":
    predict()