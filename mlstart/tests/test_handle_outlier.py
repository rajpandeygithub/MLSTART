import unittest
from mlstart.preprocessors.handle_outliers import HandleOutliers 

class TestHandleOutliers(unittest.TestCase):

    def setUp(self):
        """Set up a sample dataset for testing"""
        self.data = [
            {'age': 25, 'salary': 5000},
            {'age': 30, 'salary': 6000},
            {'age': 35, 'salary': 7000},
            {'age': 40, 'salary': 8000},
            {'age': 100, 'salary': 20000},  # outlier
            {'age': 22, 'salary': 4000},
            {'age': 18, 'salary': 3000},
            {'age': 105, 'salary': 25000},  # outlier
        ]
        self.numeric_columns = ['age', 'salary']

    """
        def test_handle_outliers(self):
            result = HandleOutliers.process(self.data, self.numeric_columns)

            # Expected result after filtering out outliers
            expected_result = [
                {'age': 25, 'salary': 5000},
                {'age': 30, 'salary': 6000},
                {'age': 35, 'salary': 7000},
                {'age': 40, 'salary': 8000},
                {'age': 22, 'salary': 4000},
                {'age': 18, 'salary': 3000},
            ]

            # Verify that the outliers (100 and 105 for age, 20000 and 25000 for salary) are removed
            self.assertEqual(result, expected_result)

    """
    def test_no_outliers(self):
        """Test when there are no outliers"""
        data_no_outliers = [
            {'age': 25, 'salary': 5000},
            {'age': 30, 'salary': 6000},
            {'age': 35, 'salary': 7000},
            {'age': 40, 'salary': 8000},
        ]
        result = HandleOutliers.process(data_no_outliers, self.numeric_columns)

        self.assertEqual(result, data_no_outliers)

    def test_single_value_column(self):
        """Test when a column contains only one unique value"""
        single_value_data = [
            {'age': 25, 'salary': 5000},
            {'age': 25, 'salary': 5000},
            {'age': 25, 'salary': 5000},
        ]
        result = HandleOutliers.process(single_value_data, self.numeric_columns)

        self.assertEqual(result, single_value_data)

    def test_outlier_in_only_one_column(self):
        """Test when outliers exist in only one column"""
        data_with_single_column_outliers = [
            {'age': 25, 'salary': 5000},
            {'age': 30, 'salary': 6000},
            {'age': 35, 'salary': 7000},
            {'age': 40, 'salary': 8000},
            {'age': 100, 'salary': 25000}, 
            {'age': 22, 'salary': 4000},
        ]
        result = HandleOutliers.process(data_with_single_column_outliers, ['salary'])

        # Verify that only the outliers in 'salary' are removed (age column is unchanged)
        expected_result = [
            {'age': 25, 'salary': 5000},
            {'age': 30, 'salary': 6000},
            {'age': 35, 'salary': 7000},
            {'age': 40, 'salary': 8000},
            {'age': 22, 'salary': 4000},
        ]
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()
