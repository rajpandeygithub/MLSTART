import unittest
from mlstart.preprocessors.remove_invalid_columns import RemoveInvalidColumns

class TestRemoveInvalidColumns(unittest.TestCase):

    def setUp(self):
        """Set up a sample dataset for testing"""
        self.data = [
            {'age': 25, 'salary': 5000, 'name': 'Alice', '': 'Invalid'},
            {'age': 30, 'salary': 6000, 'name': 'Bob', '': 'Invalid'},
            {'age': 35, 'salary': 7000, 'name': 'Charlie', '': 'Invalid'},
        ]

if __name__ == '__main__':
    unittest.main()
