import re

class NormalizeMissingValues:
    """
    Normalizes missing values and invalid entries in the dataset by replacing them with None.
    Handles numeric and categorical columns separately to ensure consistent data types.
    """

    @staticmethod
    def process(data, numeric_columns, categorical_columns):
        """
        Normalize missing values and invalid entries in the dataset by replacing them with None.

        :param data: A list of dictionaries representing the dataset.
        :param numeric_columns: A list of column names that are numeric.
        :param categorical_columns: A list of column names that are categorical.
        :return: A list of dictionaries with missing and invalid values normalized.
        """
        for row in data:
            for key, value in row.items():
                # Normalize missing values
                if value is None or (isinstance(value, str) and value.strip().lower() in ("null", "nan", "?", "", "none")):
                    row[key] = None
                elif key in numeric_columns:
                    # Validate numeric columns: ensure the value is numeric
                    try:
                        float(value)  # Check if the value can be cast to a float
                    except ValueError:
                        row[key] = None  # Replace invalid numeric value with None
                elif key in categorical_columns:
                    # Validate categorical columns
                    if not isinstance(value, (str, int)) or not str(value).strip():
                        row[key] = None  # Replace empty or non-string/int values with None
                    else:
                        # Check if all characters are invalid
                        original_length = len(str(value).strip())
                        cleaned_value = re.sub(r'[a-zA-Z0-9]', '', str(value).strip())  # Remove valid characters
                        if len(cleaned_value) == original_length:  # If all characters are invalid
                            row[key] = None

        return data
