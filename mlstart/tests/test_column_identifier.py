import unittest
from mlstart.core.column_identifier import ColumnIdentifier

class TestColumnIdentifier(unittest.TestCase):

    def test_mixed_columns(self):
        """Test dataset with mixed numeric and categorical columns."""
        data = [
            {"age": "25", "salary": "50000", "city": "Boston"},
            {"age": "30", "salary": "60000", "city": "Seattle"},
        ]
        column_identifier = ColumnIdentifier(data)
        numeric_columns, categorical_columns = column_identifier.identify_column_types()

        self.assertEqual(numeric_columns, ["age", "salary"])
        self.assertEqual(categorical_columns, ["city"])

    def test_all_numeric_columns(self):
        """Test dataset with only numeric columns."""
        data = [
            {"col1": "10", "col2": "20", "col3": "30"},
            {"col1": "40", "col2": "50", "col3": "60"},
        ]
        column_identifier = ColumnIdentifier(data)
        numeric_columns, categorical_columns = column_identifier.identify_column_types()

        self.assertEqual(numeric_columns, ["col1", "col2", "col3"])
        self.assertEqual(categorical_columns, [])

    def test_all_categorical_columns(self):
        """Test dataset with only categorical columns."""
        data = [
            {"name": "Alice", "city": "Boston", "country": "USA"},
            {"name": "Bob", "city": "Seattle", "country": "USA"},
        ]
        column_identifier = ColumnIdentifier(data)
        numeric_columns, categorical_columns = column_identifier.identify_column_types()

        self.assertEqual(numeric_columns, [])
        self.assertEqual(categorical_columns, ["name", "city", "country"])

    def test_empty_dataset(self):
        """Test behavior when the dataset is empty."""
        data = []
        column_identifier = ColumnIdentifier(data)
        with self.assertRaises(ValueError) as context:
            column_identifier.identify_column_types()
        self.assertIn("Dataset is empty or improperly formatted", str(context.exception))

    def test_missing_values(self):
        """Test dataset with missing values."""
        data = [
            {"age": "25", "salary": None, "city": "Boston"},
            {"age": "30", "salary": "60000", "city": None},
        ]
        column_identifier = ColumnIdentifier(data)
        numeric_columns, categorical_columns = column_identifier.identify_column_types()

        self.assertEqual(numeric_columns, ["age", "salary"])
        self.assertEqual(categorical_columns, ["city"])

    def test_blank_column_names(self):
        """Test dataset with blank or empty column names."""
        data = [
            {" ": "25", "salary": "50000", "city": "Boston"},
            {" ": "30", "salary": "60000", "city": "Seattle"},
        ]
        column_identifier = ColumnIdentifier(data)
        numeric_columns, categorical_columns = column_identifier.identify_column_types()

        self.assertEqual(numeric_columns, ["salary"])
        self.assertEqual(categorical_columns, ["city"])

    def test_partial_numeric_column(self):
        """Test dataset with a partially numeric column."""
        data = [
            {"age": "25", "salary": "50000", "city": "Boston"},
            {"age": "thirty", "salary": "60000", "city": "Seattle"},
        ]
        column_identifier = ColumnIdentifier(data)
        numeric_columns, categorical_columns = column_identifier.identify_column_types()

        self.assertEqual(numeric_columns, ["age", "salary"])
        self.assertEqual(categorical_columns, ["city"])

    def test_improper_format(self):
        """Test behavior with improperly formatted data."""
        data = "Not a list of dictionaries"
        column_identifier = ColumnIdentifier(data)
        with self.assertRaises(ValueError) as context:
            column_identifier.identify_column_types()
        self.assertIn("Dataset is empty or improperly formatted", str(context.exception))

if __name__ == "__main__":
    unittest.main()
