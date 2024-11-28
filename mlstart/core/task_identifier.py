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

        # Extract target column values
        target_values = [row[self.target_column] for row in self.data if row[self.target_column].strip()]
        unique_values = set(target_values)

        # Classification if unique values are small (≤10); otherwise regression
        if len(unique_values) < 10:
            self.task_type = "classification"
        else:
            self.task_type = "regression"

        return self.task_type