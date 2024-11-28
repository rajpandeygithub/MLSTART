class RemoveInvalidColumns:
    """
    Removes columns with empty names or invalid data from the dataset.
    """

    @staticmethod
    def process(data):
        """
        Remove columns with empty names or invalid data from the dataset.

        :param data: A list of dictionaries representing the dataset.
        :return: A list of dictionaries with invalid columns removed.
        """
        valid_columns = [col for col in data[0].keys() if col.strip()]

        # Remove invalid columns and return the cleaned dataset
        cleaned_data = [{col: row[col] for col in valid_columns} for row in data]
        return cleaned_data
