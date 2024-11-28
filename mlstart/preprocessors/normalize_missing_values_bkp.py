class NormalizeMissingValues:
    """
    Normalizes missing values in the dataset by replacing common missing value representations
    (e.g., 'NULL', 'NaN', '?', blank strings) with None.
    """

    @staticmethod
    def process(data, numeric_columns):
        """
        Normalize missing values in the dataset by replacing invalid or missing entries with None.

        :param data: A list of dictionaries representing the dataset.
        :param numeric_columns: A list of column names that are numeric.
        :return: A list of dictionaries with missing values normalized.
        """
        for row in data:
            for key, value in row.items():
                if value is None or (isinstance(value, str) and value.strip().lower() in ("null", "nan", "?", "")):
                    row[key] = None
                elif key in numeric_columns:
                    try:
                        float(value)
                    except ValueError:
                        row[key] = None
        return data
