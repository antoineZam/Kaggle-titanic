import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def extract_title(name):
    title = name.split(',')[1].split('.')[0].strip()
    return title


def group_titles(df):
    Officer = ['Capt', 'Col', 'Major', 'Dr', 'Rev']
    Royalty = ['Don', 'Lady', 'Countess', 'Jonkheer']
    Miss = ['Mlle', 'Ms']
    Mrs = ['Mme']
    Mr = ['Mr']
    df['Title'] = df['Title'].replace(Officer, 'Officer')
    df['Title'] = df['Title'].replace(Royalty, 'Royalty')
    df['Title'] = df['Title'].replace(Miss, 'Miss')
    df['Title'] = df['Title'].replace(Mrs, 'Mrs')
    df['Title'] = df['Title'].replace(Mr, 'Mr')
    return df


def impute_age(df):
    for title in df['Title'].unique():
        for pclass in df['Pclass'].unique():
            median_age = df[(df['Title'] == title) & (df['Pclass'] == pclass)]['Age'].median()
            df.loc[(df['Age'].isnull()) & (df['Title'] == title) & (df['Pclass'] == pclass), 'Age'] = median_age
    return df


def engineer_features(df):
    # Extract and group titles
    df['Title'] = df['Name'].apply(extract_title)
    df = group_titles(df)

    # Create age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 50, 80], labels=['Child', 'Teen', 'Young', 'Adult', 'Elder'])

    # Create fare categories
    df['FareCategory'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Med', 'Med-High', 'High'])

    # Create is_alone feature
    df['IsAlone'] = (df['SibSp'] + df['Parch'] == 0).astype(int)

    # Create fare per person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['FarePerPerson'].replace([np.inf, -np.inf], 0, inplace=True)

    return df

def handle_cabin(df):
    df['CabinKnown'] = df['Cabin'].notna().astype(int)
    return df


def handle_deck(df):
    df['Deck'] = df['Cabin'].str[0].fillna('U') # 'U' for unknown
    return df


def pclass_age_group_interaction(df):
    df['Pclass_AgeGroup'] = df['Pclass'].astype(str) + '_' + df['AgeGroup'].astype(str)
    return df


def load_data():
    train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test.csv")

    # Create 'FamilySize' feature
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

    # Engineer additional features (this depends on 'FamilySize')
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    print(train_df.isnull().sum())

    # Impute age (this depends on 'Title' created in engineer_features)
    train_df = impute_age(train_df)
    test_df = impute_age(test_df)

    train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
    test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)

    # Fill missing 'Fare' in test set with median
    test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

    # Convert 'Sex' to numerical values
    train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
    test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
    
    train_df = handle_cabin(train_df)
    test_df = handle_cabin(test_df)
    
    train_df = handle_deck(train_df)
    test_df = handle_deck(test_df)

    train_df = pclass_age_group_interaction(train_df)
    test_df = pclass_age_group_interaction(test_df)
    
 # One-Hot Encode Categorical Features
    categorical_features = ['Title', 'AgeGroup', 'FareCategory', 'Embarked', 'Deck', 'Pclass_AgeGroup']
    train_categorical = pd.get_dummies(train_df[categorical_features], drop_first=True) # drop_first=True to avoid multicollinearity
    test_categorical = pd.get_dummies(test_df[categorical_features], drop_first=True)

    # Align columns between train and test sets
    train_cols = set(train_categorical.columns)
    test_cols = set(test_categorical.columns)

    missing_in_test = list(train_cols - test_cols)
    for col in missing_in_test:
        test_categorical[col] = 0

    missing_in_train = list(test_cols - train_cols)
    for col in missing_in_train:
        train_categorical[col] = 0

    test_categorical = test_categorical[train_categorical.columns] # Ensure same order

    numerical_features = ["Pclass", "Sex", "Age", "Fare", "FamilySize",
                          "IsAlone", "FarePerPerson", "CabinKnown"]

    X_train_numerical = train_df[numerical_features]
    X_test_numerical = test_df[numerical_features]

    X_train = pd.concat([X_train_numerical, train_categorical], axis=1)
    y_train = train_df["Survived"]
    X_test = pd.concat([X_test_numerical, test_categorical], axis=1)

    # Fill any remaining NaN values with the median of the column (important after one-hot encoding too)
    for col in X_train.columns:
        if X_train[col].isnull().any():
            median_val = X_train[col].median()
            X_train[col].fillna(median_val, inplace=True)

    for col in X_test.columns:
        if X_test[col].isnull().any():
            median_val = X_test[col].median()
            X_test[col].fillna(median_val, inplace=True)

    # Save processed datasets to CSV files
    import os
    os.makedirs('../data/processed', exist_ok=True)
    X_train.to_csv('../data/processed/train_features.csv', index=False)
    y_train.to_csv('../data/processed/train_target.csv', index=False)
    X_test.to_csv('../data/processed/test_features.csv', index=False)

    print("\nProcessed datasets have been saved to:")
    print("- ../data/processed/train_features.csv")
    print("- ../data/processed/train_target.csv")
    print("- ../data/processed/test_features.csv")

    return X_train, y_train, X_test, test_df['PassengerId']


if __name__ == "__main__":
    X_train, y_train, X_test, passenger_ids = load_data()
    print("Data loaded successfully!")