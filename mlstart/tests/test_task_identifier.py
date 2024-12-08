import unittest
from mlstart.core.task_identifier import TaskIdentifier

class TaskIdentifier:
    """
    A class to determine whether the task is classification or regression
    based on the target column in the dataset.
    """

    def __init__(self, data, headers, target_column):
        """
        Initialize the TaskIdentifier with the dataset and target column.

        :param data: A list of dictionaries representing the loaded dataset.
        :param headers: A list of column headers in the dataset.
        :param target_column: A string representing the name of the target column.
        :raises ValueError: If the target column is not specified or invalid.
        """
        self.data = data
        self.headers = headers
        self.target_column = target_column
        self.task_type = None

    def determine_task_type(self):
        """
        Determine whether the task is classification or regression based on the target column.

        :return: A string indicating the task type: 'classification' or 'regression'.
        :raises ValueError: If the target column is not found in the dataset.
        """
        if self.target_column not in self.headers:
            raise ValueError(f"Target column '{self.target_column}' not found in the dataset.")

        # Extract target column values, ignoring empty or None values
        target_values = [
            row[self.target_column] for row in self.data if row.get(self.target_column) not in [None, '', ' ']
        ]

        # If target values are empty after cleaning, raise an error
        if not target_values:
            raise ValueError(f"Target column '{self.target_column}' contains only empty or None values.")

        unique_values = set(target_values)

        # If the target column has only a few unique values (<=10), treat as classification
        if len(unique_values) <= 10:
            self.task_type = "classification"
        else:
            self.task_type = "regression"

        return self.task_type


class TestTaskIdentifier(unittest.TestCase):

    def test_classification_task(self):
        data = [
            {"feature1": 1, "feature2": 2, "target": "A"},
            {"feature1": 2, "feature2": 3, "target": "B"},
            {"feature1": 3, "feature2": 4, "target": "A"},
            {"feature1": 4, "feature2": 5, "target": "B"},
            {"feature1": 5, "feature2": 6, "target": "C"}
        ]
        headers = ["feature1", "feature2", "target"]
        task_identifier = TaskIdentifier(data, headers, "target")
        self.assertEqual(task_identifier.determine_task_type(), "classification")

    def test_classification_with_numeric_values(self):
        data = [
            {"feature1": 1, "feature2": 2, "target": 1},
            {"feature1": 2, "feature2": 3, "target": 2},
            {"feature1": 3, "feature2": 4, "target": 3}
        ]
        headers = ["feature1", "feature2", "target"]
        task_identifier = TaskIdentifier(data, headers, "target")
        self.assertEqual(task_identifier.determine_task_type(), "classification")

    def test_target_column_with_few_unique_values(self):
        data = [
            {"feature1": 1, "feature2": 2, "target": 1},
            {"feature1": 2, "feature2": 3, "target": 2},
            {"feature1": 3, "feature2": 4, "target": 3},
            {"feature1": 4, "feature2": 5, "target": 4},
            {"feature1": 5, "feature2": 6, "target": 5},
            {"feature1": 6, "feature2": 7, "target": 6},
            {"feature1": 7, "feature2": 8, "target": 7},
            {"feature1": 8, "feature2": 9, "target": 8},
            {"feature1": 9, "feature2": 10, "target": 9},
            {"feature1": 10, "feature2": 11, "target": 10}
        ]
        headers = ["feature1", "feature2", "target"]
        task_identifier = TaskIdentifier(data, headers, "target")
        self.assertEqual(task_identifier.determine_task_type(), "classification")

    def test_target_column_with_mixed_data(self):
        data = [
            {"feature1": 1, "feature2": 2, "target": "A"},
            {"feature1": 2, "feature2": 3, "target": 1},
            {"feature1": 3, "feature2": 4, "target": "B"}
        ]
        headers = ["feature1", "feature2", "target"]
        task_identifier = TaskIdentifier(data, headers, "target")
        self.assertEqual(task_identifier.determine_task_type(), "classification")

if __name__ == "__main__":
    unittest.main()
