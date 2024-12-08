import unittest
from mlstart.preprocessors.normalize_missing_values import NormalizeMissingValues

class TestNormalizeMissingValues(unittest.TestCase):
    def setUp(self):
        self.data = [
            {'age': '25', 'salary': '5000', 'name': 'Alice'},
            {'age': '30', 'salary': 'NaN', 'name': 'Bob'},
            {'age': 'null', 'salary': '6000', 'name': ''},
            {'age': '40', 'salary': '7000', 'name': 'Charlie'},
            {'age': None, 'salary': '8000', 'name': 'David'},
            {'age': 'NaN', 'salary': '?', 'name': 'Eve'},
            {'age': '35', 'salary': '10000', 'name': 'Frank'},
            {'age': 'invalid', 'salary': '20000', 'name': 'Grace'},
            {'age': '50', 'salary': 'None', 'name': 'Hannah'}
        ]
        self.numeric_columns = ['age', 'salary']
        self.categorical_columns = ['name']

    def test_empty_data(self):
        result = NormalizeMissingValues.process([], self.numeric_columns, self.categorical_columns)
        self.assertEqual(result, [])



if __name__ == '__main__':
    unittest.main()
