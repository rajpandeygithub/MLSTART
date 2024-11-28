class EncodeCategorical:
    """
    Encodes categorical features using Label Encoding or One-Hot Encoding.
    """

    @staticmethod
    def process(data, categorical_columns, task_type):
        """
        Encode categorical features in the dataset.

        :param data: A list of dictionaries representing the dataset.
        :param categorical_columns: A list of column names that are categorical.
        :param task_type: A string indicating the type of task, either 'classification' or 'regression'.
        :return: A list of dictionaries with categorical features encoded.
        """
        if task_type == "classification":
            # Apply Label Encoding for classification tasks
            for column in categorical_columns:
                unique_values = list(set(row[column] for row in data if row[column] is not None))
                mapping = {val: idx for idx, val in enumerate(unique_values)}
                for row in data:
                    row[column] = mapping[row[column]]
        else:
            # Apply One-Hot Encoding for regression tasks
            for column in categorical_columns:
                unique_values = list(set(row[column] for row in data if row[column] is not None))
                for val in unique_values:
                    new_column_name = f"{column}_{val}"
                    for row in data:
                        row[new_column_name] = 1 if row[column] == val else 0
                for row in data:
                    row.pop(column, None)
        return data
