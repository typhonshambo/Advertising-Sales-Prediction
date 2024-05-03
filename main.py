from preprocessing.data_processing import load_data, preprocess_data, split_data
from models.model_building import build_linear_regression_model
from evaluation.evaluation import evaluate_model
from visualisation.visualisation import plot_predictions

def main():
    # Load data
    file_path = 'data/Advertising Budget and Sales.csv'  
    data = load_data(file_path)

    # Preprocess data
    X, y = preprocess_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Build linear regression model
    model = build_linear_regression_model()

    # Train model
    model.fit(X_train, y_train)

    # Evaluate model
    mse, r_squared = evaluate_model(model, X_test, y_test)
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r_squared}')

    # Visualize predictions
    plot_predictions(y_test, model.predict(X_test))

if __name__ == "__main__":
    main()
