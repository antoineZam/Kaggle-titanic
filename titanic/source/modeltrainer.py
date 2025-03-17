import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pandas as pd
from datasetloader import load_data

def train_models():
    # Load and preprocess data
    X_train, y_train, X_test, passenger_ids = load_data()
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42)
    }
    
    # Train and evaluate each model
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train on full training set
        model.fit(X_train_scaled, y_train)
        
        # Evaluate on training set
        train_pred = model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"Training accuracy: {train_acc:.3f}")
        
        # Save best model
        if train_acc > best_score:
            best_score = train_acc
            best_model = model
    
    # Save the best model and scaler
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print(f"\nBest model saved as 'best_model.pkl' with accuracy: {best_score:.3f}")
    
    # Generate predictions for test set
    test_pred = best_model.predict(X_test_scaled)
    
    # Create submission file
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': test_pred
    })
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file created as 'submission.csv'")
    
    return best_model

if __name__ == "__main__":
    train_models()