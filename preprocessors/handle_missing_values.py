from collections import Counter

class HandleMissingValues:
    """
    Handles missing values in numeric and categorical columns.
    """

    @staticmethod
    def process(data, numeric_columns, categorical_columns):
        """
        Handle missing values in the dataset by imputing appropriate values.

        :param data: A list of dictionaries representing the dataset.
        :param numeric_columns: A list of column names that are numeric.
        :param categorical_columns: A list of column names that are categorical.
        :return: A list of dictionaries with missing values handled.
        """
        # Handle missing values in numeric columns using the mea
        for column in numeric_columns:
            values = [float(row[column]) for row in data if row[column] is not None]
            mean_value = sum(values) / len(values)
            for row in data:
                if row[column] is None:
                    row[column] = mean_value

        # Handle missing values in categorical columns using the mode
        for column in categorical_columns:
            values = [row[column] for row in data if row[column] is not None]
            mode_value = Counter(values).most_common(1)[0][0]
            for row in data:
                if row[column] is None:
                    row[column] = mode_value

        return data
