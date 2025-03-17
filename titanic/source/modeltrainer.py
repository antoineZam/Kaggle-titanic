import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pandas as pd
from datasetloader import load_data
import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_pred_proba, title, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def train_models():
    # Create necessary directories
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../processed', exist_ok=True)
    
    # Load and preprocess data
    X_train, y_train, X_test, passenger_ids = load_data()
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grids for each model
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 4, 5]
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5]
        }
    }
    
    # Define base models
    base_models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42)
    }
    
    # Train and evaluate each model
    best_model = None
    best_score = 0
    best_model_name = None
    best_params = None
    
    # Use stratified k-fold for better handling of imbalanced data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Create a DataFrame to store model performance metrics
    model_metrics = pd.DataFrame(columns=['Model', 'Best Parameters', 'CV Score', 'Training Accuracy'])
    
    for name, model in base_models.items():
        print(f"\nTraining {name}...")
        
        # Perform Grid Search
        grid_search = GridSearchCV(
            model,
            param_grids[name],
            cv=skf,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best parameters and score
        current_params = grid_search.best_params_
        current_score = grid_search.best_score_
        current_model = grid_search.best_estimator_
        
        print(f"Best parameters: {current_params}")
        print(f"Best cross-validation score: {current_score:.3f}")
        
        # Train final model with best parameters
        current_model.fit(X_train_scaled, y_train)
        
        # Evaluate on training set
        train_pred = current_model.predict(X_train_scaled)
        train_pred_proba = current_model.predict_proba(X_train_scaled)[:, 1]
        train_acc = accuracy_score(y_train, train_pred)
        
        print(f"Training accuracy: {train_acc:.3f}")
        
        # Generate classification report
        print("\nClassification Report:")
        print(classification_report(y_train, train_pred))
        
        # Plot confusion matrix and ROC curve
        plot_confusion_matrix(
            y_train, train_pred,
            f'Confusion Matrix - {name}',
            f'../models/confusion_matrix_{name.lower().replace(" ", "_")}.png'
        )
        
        plot_roc_curve(
            y_train, train_pred_proba,
            f'ROC Curve - {name}',
            f'../models/roc_curve_{name.lower().replace(" ", "_")}.png'
        )
        
        # Save model metrics
        model_metrics = pd.concat([model_metrics, pd.DataFrame({
            'Model': [name],
            'Best Parameters': [str(current_params)],
            'CV Score': [current_score],
            'Training Accuracy': [train_acc]
        })], ignore_index=True)
        
        # Save model with its parameters
        model_info = {
            'model': current_model,
            'parameters': current_params,
            'cv_score': current_score,
            'train_accuracy': train_acc
        }
        joblib.dump(model_info, f'../models/{name.lower().replace(" ", "_")}_model.pkl')
        
        # Update best model if current one is better
        if current_score > best_score:
            best_score = current_score
            best_model = current_model
            best_model_name = name
            best_params = current_params
    
    # Save model comparison metrics
    model_metrics.to_csv('../models/model_comparison.csv', index=False)
    print("\nModel comparison metrics saved to 'models/model_comparison.csv'")
    
    # Save the best model and scaler
    joblib.dump(best_model, '../models/best_model.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')
    print(f"\nBest model ({best_model_name}) saved as 'models/best_model.pkl' with CV score: {best_score:.3f}")
    print(f"Best parameters: {best_params}")
    
    # Generate predictions for test set
    test_pred = best_model.predict(X_test_scaled)
    
    # Create submission file
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': test_pred
    })
    submission.to_csv('../submission.csv', index=False)
    print("\nSubmission file created as 'submission.csv'")
    
    return best_model

if __name__ == "__main__":
    train_models()