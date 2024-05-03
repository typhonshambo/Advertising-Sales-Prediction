import matplotlib.pyplot as plt

def plot_predictions(y_test, y_pred):
    """
    Plot actual vs. predicted sales.
    """
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Sales ($)')
    plt.ylabel('Predicted Sales ($)')
    plt.title('Actual vs. Predicted Sales')
    plt.show()
