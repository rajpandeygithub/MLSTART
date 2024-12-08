import unittest
from sklearn.model_selection import train_test_split
from mlstart.processing.datahandler import DataHandler

class TestDataHandler(unittest.TestCase):

    def setUp(self):
        """Set up a sample dataset for testing"""
        self.data = [
            {'age': 25, 'salary': 5000, 'target': 0},
            {'age': 30, 'salary': 6000, 'target': 1},
            {'age': 35, 'salary': 7000, 'target': 0},
            {'age': 40, 'salary': 8000, 'target': 1},
            {'age': 45, 'salary': 9000, 'target': 0},
        ]
        self.target_column = 'target'

    def test_split_features_and_target_classification(self):
        """Test splitting features and target for classification task"""
        handler = DataHandler(self.data, self.target_column, task_type='classification')
        X, y = handler.split_features_and_target()

        # Verify the feature matrix X and target vector y
        self.assertEqual(len(X), 5)  # 5 rows
        self.assertEqual(len(y), 5)  # 5 rows

        # Check that the target values are integers for classification
        self.assertTrue(all(isinstance(val, int) for val in y))

        # Check that the feature matrix contains only numeric values
        for row in X:
            self.assertTrue(all(isinstance(val, float) for val in row))

    def test_split_features_and_target_regression(self):
        """Test splitting features and target for regression task"""
        handler = DataHandler(self.data, self.target_column, task_type='regression')
        X, y = handler.split_features_and_target()

        # Verify the feature matrix X and target vector y
        self.assertEqual(len(X), 5)  # 5 rows
        self.assertEqual(len(y), 5)  # 5 rows

        # Check that the target values are floats for regression
        self.assertTrue(all(isinstance(val, float) for val in y))

        # Check that the feature matrix contains only numeric values
        for row in X:
            self.assertTrue(all(isinstance(val, float) for val in row))

    def test_invalid_task_type(self):
        """Test invalid task type raises a ValueError"""
        handler = DataHandler(self.data, self.target_column, task_type='invalid')

        with self.assertRaises(ValueError):
            handler.split_features_and_target()

    def test_train_test_splitting(self):
        """Test splitting data into train and test sets"""
        handler = DataHandler(self.data, self.target_column, task_type='classification')
        X_train, X_test, y_train, y_test = handler.train_test_splitting(test_size=0.4)

        # Verify that the train and test sets are split correctly
        self.assertEqual(len(X_train), 3)  # 60% of the data
        self.assertEqual(len(X_test), 2)   # 40% of the data

        # Check that the length of X_train, X_test, y_train, y_test match
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))

        # Check that the target values in y_train and y_test match the correct split
        for y_value in y_train + y_test:
            self.assertIn(y_value, [0, 1])

    def test_train_test_splitting_default(self):
        """Test train/test split with default parameters (80% train, 20% test)"""
        handler = DataHandler(self.data, self.target_column, task_type='classification')
        X_train, X_test, y_train, y_test = handler.train_test_splitting()

        # Verify that the default split is 80/20
        self.assertEqual(len(X_train), 4)  # 80% of the data
        self.assertEqual(len(X_test), 1)   # 20% of the data

    def test_train_test_splitting_random_state(self):
        """Test reproducibility of the train/test split using random_state"""
        handler = DataHandler(self.data, self.target_column, task_type='classification')
        X_train1, X_test1, y_train1, y_test1 = handler.train_test_splitting(random_state=42)
        X_train2, X_test2, y_train2, y_test2 = handler.train_test_splitting(random_state=42)

        # Ensure that the splits are the same across multiple runs with the same random state
        self.assertEqual(X_train1, X_train2)
        self.assertEqual(X_test1, X_test2)
        self.assertEqual(y_train1, y_train2)
        self.assertEqual(y_test1, y_test2)

if __name__ == '__main__':
    unittest.main()
