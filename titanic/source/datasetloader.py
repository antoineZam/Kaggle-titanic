import pandas as pd
from sklearn.preprocessing import LabelEncoder


def extract_title(name):
    title = name.split(',')[1].split('.')[0].strip()
    return title


def impute_age(df):
    for title in df['Title'].unique():
        for pclass in df['Pclass'].unique():
            median_age = df[(df['Title'] == title) & (df['Pclass'] == pclass)]['Age'].median()
            df.loc[(df['Age'].isnull()) & (df['Title'] == title) & (df['Pclass'] == pclass), 'Age'] = median_age
    return df


def engineer_features(df):
    # Extract titles
    df['Title'] = df['Name'].apply(extract_title)
    
    # Create age groups
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 50, 80], labels=['Child', 'Teen', 'Young', 'Adult', 'Elder'])
    
    # Create fare categories
    df['FareCategory'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Med', 'Med-High', 'High'])
    
    # Create is_alone feature
    df['IsAlone'] = (df['SibSp'] + df['Parch'] == 0).astype(int)
    
    # Create fare per person
    df['FarePerPerson'] = df['Fare'] / (df['FamilySize'])
    
    return df


def load_data():
    train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test.csv")

    #print(train_df.head())
    #print(test_df.head())

    print(train_df.isnull().sum())

    train_df = impute_age(train_df)
    test_df = impute_age(test_df)

    train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
    test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)

    # Fill missing 'Fare' in test set with median
    test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

    # Convert 'Sex' to numerical values
    train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
    test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

    # Convert 'Embarked' to numerical using LabelEncoder
    encoder = LabelEncoder()
    train_df['Embarked'] = encoder.fit_transform(train_df['Embarked'])
    test_df['Embarked'] = encoder.transform(test_df['Embarked'])

    # Create 'FamilySize' feature
    train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
    test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

    # Engineer additional features
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", 
                "IsAlone", "FarePerPerson"]

    X_train = train_df[features]
    y_train = train_df["Survived"]
    X_test = test_df[features]

    # Save processed datasets to CSV files
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