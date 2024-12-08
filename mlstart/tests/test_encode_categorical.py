import unittest
from mlstart.preprocessors.encode_categorical import EncodeCategorical

class TestEncodeCategorical(unittest.TestCase):

    def test_one_hot_encoding_regression(self):
        # Sample data for regression (task_type = 'regression')
        data = [
            {"color": "red", "size": "small"},
            {"color": "blue", "size": "large"},
            {"color": "red", "size": "medium"},
            {"color": "green", "size": "small"}
        ]
        categorical_columns = ["color", "size"]
        task_type = "regression"
        
        # Apply one-hot encoding
        encoded_data = EncodeCategorical.process(data, categorical_columns, task_type)
        
        # Expected output after one-hot encoding (regression)
        expected_data = [
            {"color_red": 1, "color_blue": 0, "color_green": 0, "size_small": 1, "size_large": 0, "size_medium": 0},
            {"color_red": 0, "color_blue": 1, "color_green": 0, "size_small": 0, "size_large": 1, "size_medium": 0},
            {"color_red": 1, "color_blue": 0, "color_green": 0, "size_small": 0, "size_large": 0, "size_medium": 1},
            {"color_red": 0, "color_blue": 0, "color_green": 1, "size_small": 1, "size_large": 0, "size_medium": 0}
        ]
        
        # Assertions for one-hot encoding (regression)
        self.assertEqual(encoded_data, expected_data)

    def test_no_categorical_columns(self):
        # Sample data with no categorical columns
        data = [{"color": "red", "size": "small"}, {"color": "blue", "size": "large"}]
        categorical_columns = []
        task_type = "classification"
        
        # Apply encoding with no categorical columns
        encoded_data = EncodeCategorical.process(data, categorical_columns, task_type)
        
        # Expected result is the same data as no encoding is done
        expected_data = [{"color": "red", "size": "small"}, {"color": "blue", "size": "large"}]
        
        # Assertions for no categorical columns
        self.assertEqual(encoded_data, expected_data)

if __name__ == "__main__":
    unittest.main()
