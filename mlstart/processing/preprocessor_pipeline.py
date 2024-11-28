class PreprocessorPipeline:
    """
    Orchestrates the preprocessing pipeline.
    """

    def __init__(self, task_type):
        """
        Initializes the PreprocessorPipeline class.

        :param task_type: A string indicating the task type ('classification' or 'regression').
        """
        self.task_type = task_type

    def run(self, data, numeric_columns, categorical_columns):
        """
        Executes the preprocessing pipeline in a sequential manner.

        :param data: A list of dictionaries representing the dataset.
        :param numeric_columns: A list of column names identified as numeric.
        :param categorical_columns: A list of column names identified as categorical.
        :return: A preprocessed dataset as a list of dictionaries.
        """
        from mlstart.preprocessors.remove_invalid_columns import RemoveInvalidColumns
        from mlstart.preprocessors.normalize_missing_values import NormalizeMissingValues
        from mlstart.preprocessors.handle_missing_values import HandleMissingValues
        from mlstart.preprocessors.encode_categorical import EncodeCategorical
        from mlstart.preprocessors.scale_numeric import ScaleNumeric
        from mlstart.preprocessors.remove_duplicates import RemoveDuplicates
        from mlstart.preprocessors.handle_outliers import HandleOutliers

        data = RemoveInvalidColumns.process(data)
        data = NormalizeMissingValues.process(data, numeric_columns,categorical_columns)
        data = HandleMissingValues.process(data, numeric_columns, categorical_columns)
        data = EncodeCategorical.process(data, categorical_columns, self.task_type)
        data = ScaleNumeric.process(data, numeric_columns)
        data = RemoveDuplicates.process(data)
        if self.task_type == "regression":
            data = HandleOutliers.process(data, numeric_columns)

        return data
