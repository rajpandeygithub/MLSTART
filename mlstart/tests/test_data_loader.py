import unittest
from mlstart.core.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def test_valid_file(self):
        """Test loading a valid CSV file."""
        data_loader = DataLoader("data/auto_mpg.csv")
        headers, data = data_loader.load_data()
        expected_headers = ["mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin","car name"] 
        self.assertEqual(headers, expected_headers)
        self.assertGreater(len(data), 0)

    def test_file_not_found(self):
        """Test behavior when the file does not exist."""
        with self.assertRaises(FileNotFoundError):
            DataLoader("data/non_existent_file.csv").load_data()

    def test_empty_file(self):
        """Test behavior with an empty CSV file."""
        with self.assertRaises(ValueError) as context:
            DataLoader("data/empty.csv").load_data()
        self.assertIn("The CSV file is empty", str(context.exception))


    def test_non_csv_file(self):
        """Test behavior with a non-CSV file format."""
        with self.assertRaises(ValueError) as context:
            DataLoader("data/non_csv.txt").load_data()
        self.assertIn("Error loading file", str(context.exception))

    def test_partially_valid_file(self):
        """Test behavior with a file containing mixed valid and invalid rows."""
        data_loader = DataLoader("data/partial_valid.csv")
        headers, data = data_loader.load_data()
        self.assertIsNotNone(headers)
        self.assertGreater(len(data), 0)  # Check that valid rows are still loaded