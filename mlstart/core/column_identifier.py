class ColumnIdentifier:
    """
    A class to identify numeric and categorical columns in a dataset.
    """

    def __init__(self, data):
        """
        Initialize the ColumnIdentifier with dataset rows.

        :param data: A list of dictionaries where each dictionary represents a row of the dataset.
        :raises ValueError: If the provided dataset is not in the correct format.
        """
        self.data = data


    def identify_column_types(self):
        """
        Identify numeric and categorical columns in the dataset.

        :return: A tuple containing:
                 - numeric_columns: A list of column names with numeric data.
                 - categorical_columns: A list of column names with categorical data.
        :raises ValueError: If the dataset is empty or improperly formatted.
        """
        # Ensure data is not empty
        if not self.data or not isinstance(self.data, list) or not self.data[0]:
            raise ValueError("Dataset is empty or improperly formatted.")

        numeric_columns = []
        categorical_columns = []

        # Filter out empty column names
        valid_columns = [col for col in self.data[0].keys() if col.strip()]

        # Loop through each valid column to determine its type
        for column in valid_columns:
            try:
                # Try casting the first non-empty, non-None value in the column to a float
                for row in self.data:
                    value = row[column]
                    if value is not None and str(value).strip():  # Check for None or blank strings
                        float(value)  # Attempt to cast
                        break
                numeric_columns.append(column)
            except ValueError:
                # If casting fails, it's a categorical column
                categorical_columns.append(column)

        return numeric_columns, categorical_columns