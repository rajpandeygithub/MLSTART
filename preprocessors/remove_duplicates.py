class RemoveDuplicates:
    """
    Removes duplicate rows from the dataset.
    """

    @staticmethod
    def process(data):
        """
        Remove duplicate rows from the dataset.

        :param data: A list of dictionaries representing the dataset.
        :return: A list of dictionaries with duplicates removed.
        """
        unique_data = []
        seen = set()
        for row in data:
            row_tuple = tuple(row.items())
            if row_tuple not in seen:
                seen.add(row_tuple)
                unique_data.append(row)
        return unique_data
