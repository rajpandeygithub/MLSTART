import csv

class DataLoader:
    """
    A class to load data from a CSV file.
    """

    def __init__(self, file_path):
        """
        Initialize the DataLoader class with the path to the dataset file.

        :param file_path: A string representing the path to the dataset file (CSV format).
        :raises ValueError: If the file path is invalid or not accessible.
        """
        self.file_path = file_path
        self.headers = None
        self.data = []

    def load_data(self):
        """
        Load the dataset from the specified file path.

        :return: A tuple containing:
                 - headers: A list of column headers in the dataset.
                 - data: A list of dictionaries where each dictionary represents a row of the dataset.
        :raises FileNotFoundError: If the file does not exist at the specified path.
        :raises ValueError: If the file is empty, not in CSV format, or improperly formatted.
        """
        try:
            with open(self.file_path, mode="r") as file:
                reader = csv.DictReader(file)
                self.headers = reader.fieldnames

                if not self.headers:
                    raise ValueError("The CSV file is empty \nor the file is not in CSV format \nor improperly formatted.")

                for row in reader:
                    self.data.append(row)

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")
        except Exception as e:
            raise ValueError(f"Error loading file: {e}")

        return self.headers, self.data
