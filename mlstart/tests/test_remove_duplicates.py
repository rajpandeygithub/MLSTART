import unittest
from mlstart.preprocessors.remove_duplicates import RemoveDuplicates

class TestRemoveDuplicates(unittest.TestCase):

    def setUp(self):
        """Set up a sample dataset for testing"""
        self.data = [
            {'age': 25, 'salary': 5000, 'name': 'Alice'},
            {'age': 30, 'salary': 6000, 'name': 'Bob'},
            {'age': 25, 'salary': 5000, 'name': 'Alice'},  
            {'age': 40, 'salary': 7000, 'name': 'Charlie'},
            {'age': 30, 'salary': 6000, 'name': 'Bob'},  
            {'age': 35, 'salary': 8000, 'name': 'David'}
        ]

    def test_remove_duplicates(self):
        """Test removing duplicate rows from the dataset"""
        result = RemoveDuplicates.process(self.data)

        # Expected result after duplicates are removed
        expected_result = [
            {'age': 25, 'salary': 5000, 'name': 'Alice'},
            {'age': 30, 'salary': 6000, 'name': 'Bob'},
            {'age': 40, 'salary': 7000, 'name': 'Charlie'},
            {'age': 35, 'salary': 8000, 'name': 'David'}
        ]

        # Verify that the duplicates are removed and only unique rows remain
        self.assertEqual(result, expected_result)

    def test_no_duplicates(self):
        """Test when there are no duplicates in the dataset"""
        data_without_duplicates = [
            {'age': 25, 'salary': 5000, 'name': 'Alice'},
            {'age': 30, 'salary': 6000, 'name': 'Bob'},
            {'age': 40, 'salary': 7000, 'name': 'Charlie'},
            {'age': 35, 'salary': 8000, 'name': 'David'}
        ]
        result = RemoveDuplicates.process(data_without_duplicates)

        # If there are no duplicates, the result should be the same as the input
        self.assertEqual(result, data_without_duplicates)

    def test_empty_data(self):
        """Test when the dataset is empty"""
        empty_data = []
        result = RemoveDuplicates.process(empty_data)
        self.assertEqual(result, empty_data)  

    def test_all_duplicates(self):
        """Test when all rows are duplicates"""
        all_duplicates = [
            {'age': 25, 'salary': 5000, 'name': 'Alice'},
            {'age': 25, 'salary': 5000, 'name': 'Alice'},
            {'age': 25, 'salary': 5000, 'name': 'Alice'}
        ]
        result = RemoveDuplicates.process(all_duplicates)

        # Only one row should remain as all rows are identical
        expected_result = [{'age': 25, 'salary': 5000, 'name': 'Alice'}]
        self.assertEqual(result, expected_result)

    def test_duplicates_with_diff_order(self):
        """Test when duplicate rows are in a different order"""
        data_with_diff_order = [
            {'age': 30, 'salary': 6000, 'name': 'Bob'},
            {'age': 40, 'salary': 7000, 'name': 'Charlie'},
            {'age': 25, 'salary': 5000, 'name': 'Alice'},
            {'age': 30, 'salary': 6000, 'name': 'Bob'},  
            {'age': 25, 'salary': 5000, 'name': 'Alice'} 
        ]
        result = RemoveDuplicates.process(data_with_diff_order)

        # Expected result after duplicates are removed
        expected_result = [
            {'age': 30, 'salary': 6000, 'name': 'Bob'},
            {'age': 40, 'salary': 7000, 'name': 'Charlie'},
            {'age': 25, 'salary': 5000, 'name': 'Alice'}
        ]
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()
