import unittest
from sklearn.dummy import DummyClassifier, DummyRegressor
from mlstart.models.model_evaluation import ModelEvaluator

class TestModelEvaluator(unittest.TestCase):

    def test_classification_metrics(self):
        # Sample data for classification
        y_true = [1, 0, 1, 1, 0, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 0, 1, 0, 1]

        # Initialize evaluator for classification task
        evaluator = ModelEvaluator(task_type="classification")

        # Calculate metrics
        accuracy = evaluator.calculate_accuracy(y_true, y_pred)
        precision = evaluator.calculate_precision(y_true, y_pred)
        recall = evaluator.calculate_recall(y_true, y_pred)
        f1_score = evaluator.calculate_f1_score(precision, recall)

        # Assertions for classification
        self.assertAlmostEqual(accuracy, 0.875)
        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 0.8)
        self.assertAlmostEqual(f1_score, 0.888888888888889, places=5) 

    def test_regression_metrics(self):
        # Sample data for regression
        y_true = [3.0, 2.5, 4.0, 5.0]
        y_pred = [2.8, 2.4, 3.9, 5.2]

        # Initialize evaluator for regression task
        evaluator = ModelEvaluator(task_type="regression")

        # Calculate metrics
        mae = evaluator.calculate_mae(y_true, y_pred)
        mse = evaluator.calculate_mse(y_true, y_pred)
        rmse = evaluator.calculate_rmse(mse)

        # Assertions for regression
        self.assertAlmostEqual(mae, 0.15)
        self.assertAlmostEqual(mse, 0.025)
        self.assertAlmostEqual(rmse, 0.15811388300841897)

    
    def test_evaluate_classification_model(self):
        # Sample data for classification
        X_test = [[0], [1], [2], [3]]
        y_test = [0, 1, 0, 1]
        
        # Create a dummy classifier model (a simple model for testing)
        model = DummyClassifier(strategy="most_frequent")
        model.fit(X_test, y_test)

        evaluator = ModelEvaluator(task_type="classification")
        metrics = evaluator.evaluate_model(model, X_test, y_test)

        # Assertions for classification metrics
        self.assertIn("Accuracy", metrics)
        self.assertIn("Precision", metrics)
        self.assertIn("Recall", metrics)
        self.assertIn("F1 Score", metrics)
        self.assertAlmostEqual(metrics["Accuracy"], 0.5)

    def test_evaluate_regression_model(self):
        # Sample data for regression
        X_test = [[0], [1], [2], [3]]
        y_test = [0.5, 1.5, 2.5, 3.5]
        
        # Create a dummy regressor model (a simple model for testing)
        model = DummyRegressor(strategy="mean")
        model.fit(X_test, y_test)

        evaluator = ModelEvaluator(task_type="regression")
        metrics = evaluator.evaluate_model(model, X_test, y_test)

        # Assertions for regression metrics
        self.assertIn("MAE", metrics)
        self.assertIn("MSE", metrics)
        self.assertIn("RMSE", metrics)
        self.assertAlmostEqual(metrics["MAE"], 1.0)

if __name__ == "__main__":
    unittest.main()
