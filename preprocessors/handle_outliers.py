class HandleOutliers:
    """
    Handles outliers in numeric features using the IQR method.
    """

    @staticmethod
    def process(data, numeric_columns):
        """
        Handle outliers in numeric columns by filtering out rows with values outside the IQR range.

        :param data: A list of dictionaries representing the dataset.
        :param numeric_columns: A list of column names that are numeric.
        :return: A list of dictionaries with outliers removed based on the IQR method.
        """
        for column in numeric_columns:
            # Extract numeric values from the column
            values = [float(row[column]) for row in data]

            # Calculate Q1, Q3, and IQR
            q1 = sorted(values)[int(len(values) * 0.25)]
            q3 = sorted(values)[int(len(values) * 0.75)]
            iqr = q3 - q1

            # Determine bounds for outliers
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            data = [row for row in data if lower_bound <= float(row[column]) <= upper_bound] # Filter rows that fall within the bounds
        return data
