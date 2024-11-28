from sklearn.model_selection import train_test_split

class DataHandler:
    """
    A class to handle data splitting and preparation for training and testing.
    """

    def __init__(self, data, target_column, task_type):
        """
        Initializes the DataHandler class.

        :param data: A list of dictionaries representing the preprocessed dataset.
        :param target_column: A string specifying the name of the target column.
        :param task_type: A string indicating the task type ('classification' or 'regression').
        """
        self.data = data
        self.target_column = target_column
        self.task_type = task_type

    def split_features_and_target(self):
        """
        Splits the dataset into features (X) and target (y).

        :return: 
            - X: A list of lists representing the feature matrix.
            - y: A list representing the target vector.
        :raises ValueError: If the task type is invalid or not recognized.
        """
        X = []
        y = []

        for row in self.data:
            # Exclude the target column from the feature set
            X.append([float(row[key]) for key in row.keys() if key != self.target_column])
            # Add the target column to the target vector
            if self.task_type == "classification":
                y.append(int(row[self.target_column]))  # Ensure target is integer for classification
            elif self.task_type == "regression":
                y.append(float(row[self.target_column]))  # Ensure target is float for regression
            else:
                raise ValueError(f"Invalid task type: {self.task_type}")

        return X, y

    def train_test_splitting(self, test_size=0.2, random_state=42):
        """
        Splits the dataset into training and testing sets.

        :param test_size: A float representing the proportion of the dataset to include in the test split. Default is 0.2.
        :param random_state: An integer seed for reproducibility of the split. Default is 42.
        :return: 
            - X_train: A list of lists representing the training feature matrix.
            - X_test: A list of lists representing the testing feature matrix.
            - y_train: A list representing the training target vector.
            - y_test: A list representing the testing target vector.
        """
        X, y = self.split_features_and_target()
        return train_test_split(X, y, test_size=test_size, random_state=random_state)