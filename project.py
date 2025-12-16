"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
- JP
- Miguel
- 
- 

Dataset: Medical Insurance Cost Prediction
Predicting: The cost of medical insurance based on various features.
Features: age, sex, BMI, children, smoker, region, charges
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# TODO: Update this with your actual filename
DATA_FILE = 'medical_insurance.csv'

FEATURE_COLUMNS = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
TARGET_COLUMN = 'charges'

def load_and_explore_data(filename):
    """
    Load your dataset and print basic information
    
    TODO:
    - Load the CSV file
    - Print the shape (rows, columns)
    - Print the first few rows
    - Print summary statistics
    - Check for missing values
    """
    print("=" * 70)
    print("LOADING AND EXPLORING DATA")
    print("=" * 70)
    
    # Your code here
    data = pd.read_csv(filename)

    print("Shape:", data.shape)
    print("\nFirst 5 rows:")
    print(data.head())
    
    print("\nSummary statistics:")
    print(data.describe())
    
    print("\nMissing values:")
    print(data.isnull().sum())

    return data


def visualize_data(data):
    """
    Create visualizations to understand your data
    
    TODO:
    - Create scatter plots for each feature vs target
    - Save the figure
    - Identify which features look most important
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
    """
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)
    
    # Your code here
    # Hint: Use subplots like in Part 2!
    
    numeric_features = ['age', 'bmi', 'children']
    
    plt.figure(figsize=(12, 4))
    for i, feature in enumerate(numeric_features):
        plt.subplot(1, 3, i + 1)
        plt.scatter(data[feature], data[TARGET_COLUMN])
        plt.xlabel(feature)
        plt.ylabel(TARGET_COLUMN)
        plt.title(f'{feature} vs Charges')
    
    plt.tight_layout()
    plt.savefig("feature_relationships.png")
    plt.show()


def prepare_and_split_data(data):
    """
    Prepare X and y, then split into train/test
    
    TODO:
    - Separate features (X) and target (y)
    - Split into train/test (80/20)
    - Print the sizes
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)
    
    # Your code here
    
    data_encoded = pd.get_dummies(
        data[FEATURE_COLUMNS + [TARGET_COLUMN]],
        drop_first=True
    )
    
    X = data_encoded.drop(TARGET_COLUMN, axis=1)
    y = data_encoded[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training set size:", X_train.shape)
    print("Test set size:", X_test.shape)
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train the linear regression model
    
    TODO:
    - Create and train a LinearRegression model
    - Print the equation with all coefficients
    - Print feature importance (rank features by coefficient magnitude)
    
    Args:
        X_train: training features
        y_train: training target
        feature_names: list of feature names
        
    Returns:
        trained model
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    # Your code here
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("Intercept:", model.intercept_)
    
    coefficients = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_
    })
    
    coefficients['Absolute Value'] = coefficients['Coefficient'].abs()
    coefficients = coefficients.sort_values(by='Absolute Value', ascending=False)
    
    print("\nFeature Importance:")
    print(coefficients[['Feature', 'Coefficient']])
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    TODO:
    - Make predictions on test set
    - Calculate R² score
    - Calculate RMSE
    - Print results clearly
    - Create a comparison table (first 10 examples)
    
    Args:
        model: trained model
        X_test: test features
        y_test: test target
        
    Returns:
        predictions
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    
    # Your code here
    
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print("R² Score:", r2)
    print("RMSE:", rmse)
    
    comparison = pd.DataFrame({
        'Actual Charges': y_test.values[:10],
        'Predicted Charges': predictions[:10]
    })
    
    print("\nComparison (first 10):")
    print(comparison)
    
    return predictions


def make_prediction(model):
    """
    Make a prediction for a new example
    
    TODO:
    - Create a sample input (you choose the values!)
    - Make a prediction
    - Print the input values and predicted output
    
    Args:
        model: trained model
        feature_names: list of feature names
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    # Your code here
    # Example: If predicting house price with [sqft, bedrooms, bathrooms]
    # sample = pd.DataFrame([[2000, 3, 2]], columns=feature_names)
    
    sample = pd.DataFrame([{
        'age': 40,
        'bmi': 30,
        'children': 2,
        'sex_male': 1,
        'smoker_yes': 1,
        'region_northwest': 0,
        'region_southeast': 1,
        'region_southwest': 0
    }])
    
    prediction = model.predict(sample)
    
    print("Sample input:")
    print(sample)
    print("Predicted insurance charge:", prediction[0])


if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data(DATA_FILE)
    
    # Step 2: Visualize
    visualize_data(data)
    
    # Step 3: Prepare and split
    X_train, X_test, y_train, y_test = prepare_and_split_data(data)
    
    # Step 4: Train
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate
    predictions = evaluate_model(model, X_test, y_test)
    
    # Step 6: Make a prediction, add features as an argument
    make_prediction(model)
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Analyze your results")
    print("2. Try improving your model (add/remove features)")
    print("3. Create your presentation")
    print("4. Practice presenting with your group!")

