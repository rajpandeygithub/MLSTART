import math
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier

class ModelTrainer:
    """
    A class to train machine learning models for classification and regression tasks.
    """

    def __init__(self, task_type):
        """
        Initialize the ModelTrainer class.

        :param task_type: A string representing the type of task, either 'classification' or 'regression'.
        :raises ValueError: If an invalid task type is provided.
        """
        self.task_type = task_type
        self.models = []  # List to store initialized models

    def initialize_models(self):
        """
        Initialize models based on the task type.

        :raises ValueError: If the task type is invalid.
        """
        if self.task_type == "classification":
            self.models = [
                ("Logistic Regression", LogisticRegression()),
                ("Decision Tree", DecisionTreeClassifier()),
                ("K-Nearest Neighbors", KNeighborsClassifier()),
            ]
        elif self.task_type == "regression":
            self.models = [
                ("Linear Regression", LinearRegression()),
                ("Ridge Regression", Ridge()),
                ("Decision Tree", DecisionTreeRegressor()),
            ]
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")

    def train_models(self, X_train, y_train):
        """
        Train the initialized models on the given training data.

        :param X_train: A list or ndarray representing the feature matrix for training.
        :param y_train: A list or ndarray representing the target vector for training.
        :return: A list of tuples where each tuple contains the model name and the trained model object.
        :raises Exception: If a model fails to train, an error message is printed.
        """
        trained_models = []
        for name, model in self.models:
            try:
                model.fit(X_train, y_train)
                trained_models.append((name, model))
                #print(f"Trained {name} successfully.")
            except Exception as e:
                print(f"Failed to train {name}: {e}")
        return trained_models