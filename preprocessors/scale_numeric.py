import math

class ScaleNumeric:
    """
    Scales numeric features using Z-score normalization.
    """

    @staticmethod
    def process(data, numeric_columns):
        """
        Scale numeric features using Z-score normalization.

        :param data: A list of dictionaries representing the dataset.
        :param numeric_columns: A list of column names that are numeric.
        :return: A list of dictionaries with numeric features scaled.
        """
        for column in numeric_columns:
            values = [float(row[column]) for row in data]
            mean_value = sum(values) / len(values)
            std_dev = math.sqrt(sum((x - mean_value) ** 2 for x in values) / len(values))
            if std_dev == 0:
                for row in data:
                    row[column] = 0.0
            else:
                for row in data:
                    row[column] = (float(row[column]) - mean_value) / std_dev
        return data
