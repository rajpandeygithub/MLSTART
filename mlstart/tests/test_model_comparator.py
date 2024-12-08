import unittest
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from mlstart.models.model_comparison import ModelComparator

class TestModelComparator(unittest.TestCase):

    def test_classification_model_comparator(self):
        """Test comparing classification models using ModelComparator."""
        # Load a sample classification dataset (Iris)
        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

        # Initialize and train models
        models = [
            ("Logistic Regression", LogisticRegression(max_iter=200)),
            ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ]
        
        # Train models
        trained_models = []
        for name, model in models:
            model.fit(X_train, y_train)
            trained_models.append((name, model))

        # Initialize ModelComparator for classification
        model_comparator = ModelComparator(task_type="classification")

        # Compare models
        best_model_name, best_model, model_metrics, total_ranks = model_comparator.compare_models(trained_models, X_test, y_test)

        # Test that the best model is returned and ranks are calculated
        self.assertIn(best_model_name, ["Logistic Regression", "Decision Tree"])
        self.assertIsNotNone(best_model)
        self.assertIn(best_model_name, total_ranks)
        
        # Test evaluation metrics (accuracy for classification)
        self.assertIn("Accuracy", model_metrics[best_model_name])  
        accuracy = model_metrics[best_model_name]["Accuracy"]  
        self.assertGreater(accuracy, 0.6) 

    def test_regression_model_comparator(self):
        """Test comparing regression models using ModelComparator."""
        # Load a sample regression dataset (California housing instead of Boston)
        data = fetch_california_housing()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

        # Initialize and train models
        models = [
            ("Linear Regression", LinearRegression()),
            ("Decision Tree", DecisionTreeRegressor(random_state=42)),
        ]
        
        # Train models
        trained_models = []
        for name, model in models:
            model.fit(X_train, y_train)
            trained_models.append((name, model))

        # Initialize ModelComparator for regression
        model_comparator = ModelComparator(task_type="regression")

        # Compare models
        best_model_name, best_model, model_metrics, total_ranks = model_comparator.compare_models(trained_models, X_test, y_test)

        # Test that the best model is returned and ranks are calculated
        self.assertIn(best_model_name, ["Linear Regression", "Decision Tree"])
        self.assertIsNotNone(best_model)
        self.assertIn(best_model_name, total_ranks)

        # Test evaluation metrics (e.g., MSE for regression)
        self.assertIn("MSE", model_metrics[best_model_name])
        mse = model_metrics[best_model_name]["MSE"]
        self.assertLess(mse, 50) 

    def test_invalid_task_type(self):
        """Test that an invalid task type raises a ValueError."""
        with self.assertRaises(ValueError):
            model_comparator = ModelComparator(task_type="invalid")
            model_comparator.compare_models([], [], [])

if __name__ == '__main__':
    unittest.main()
