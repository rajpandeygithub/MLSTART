class ModelEvaluator:
    """
    A class to evaluate machine learning models for classification and regression tasks using custom metric implementations.
    """

    def __init__(self, task_type):
        """
        Initialize the ModelEvaluator class.

        :param task_type: A string indicating the type of task, either 'classification' or 'regression'.
        :raises ValueError: If an invalid task type is provided.
        """
        self.task_type = task_type

    def calculate_accuracy(self, y_true, y_pred):
        """
        Calculate accuracy for classification tasks.

        :param y_true: A list or array of true target values.
        :param y_pred: A list or array of predicted values.
        :return: A float representing the accuracy score.
        """
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct / len(y_true)

    def calculate_precision(self, y_true, y_pred):
        """
        Calculate precision for classification tasks.

        :param y_true: A list or array of true target values.
        :param y_pred: A list or array of predicted values.
        :return: A float representing the precision score.
        """
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == pred and pred == 1)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != pred and pred == 1)
        return tp / (tp + fp) if (tp + fp) > 0 else 0

    def calculate_recall(self, y_true, y_pred):
        """
        Calculate recall for classification tasks.

        :param y_true: A list or array of true target values.
        :param y_pred: A list or array of predicted values.
        :return: A float representing the recall score.
        """
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == pred and pred == 1)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true != pred and pred == 0)
        return tp / (tp + fn) if (tp + fn) > 0 else 0

    def calculate_f1_score(self, precision, recall):
        """
        Calculate F1 score for classification tasks.

        :param precision: A float representing the precision score.
        :param recall: A float representing the recall score.
        :return: A float representing the F1 score.
        """
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def calculate_mae(self, y_true, y_pred):
        """
        Calculate Mean Absolute Error (MAE) for regression tasks.

        :param y_true: A list or array of true target values.
        :param y_pred: A list or array of predicted values.
        :return: A float representing the MAE.
        """
        return sum(abs(true - pred) for true, pred in zip(y_true, y_pred)) / len(y_true)

    def calculate_mse(self, y_true, y_pred):
        """
        Calculate Mean Squared Error (MSE) for regression tasks.

        :param y_true: A list or array of true target values.
        :param y_pred: A list or array of predicted values.
        :return: A float representing the MSE.
        """
        return sum((true - pred) ** 2 for true, pred in zip(y_true, y_pred)) / len(y_true)

    def calculate_rmse(self, mse):
        """
        Calculate Root Mean Squared Error (RMSE) for regression tasks.

        :param mse: A float representing the Mean Squared Error (MSE).
        :return: A float representing the RMSE.
        """
        return mse ** 0.5

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate a single model using appropriate metrics for classification or regression.

        :param model: The trained model to evaluate.
        :param X_test: A list or array representing the test features.
        :param y_test: A list or array representing the true test target values.
        :return: A dictionary containing evaluation metrics for the model.
        :raises ValueError: If the task type is invalid.
        """
        predictions = model.predict(X_test)
        metrics = {}

        if self.task_type == "classification":
            accuracy = self.calculate_accuracy(y_test, predictions)
            precision = self.calculate_precision(y_test, predictions)
            recall = self.calculate_recall(y_test, predictions)
            f1_score = self.calculate_f1_score(precision, recall)

            metrics["Accuracy"] = accuracy
            metrics["Precision"] = precision
            metrics["Recall"] = recall
            metrics["F1 Score"] = f1_score

        elif self.task_type == "regression":
            mae = self.calculate_mae(y_test, predictions)
            mse = self.calculate_mse(y_test, predictions)
            rmse = self.calculate_rmse(mse)

            metrics["MAE"] = mae
            metrics["MSE"] = mse
            metrics["RMSE"] = rmse

        else:
            raise ValueError(f"Invalid task type: {self.task_type}")

        return metrics
