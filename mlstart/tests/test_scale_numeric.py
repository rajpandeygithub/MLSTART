import unittest
import math
from mlstart.preprocessors.scale_numeric import ScaleNumeric

class TestScaleNumeric(unittest.TestCase):

    def setUp(self):
        """Set up a sample dataset for testing"""
        self.data = [
            {'age': 25, 'salary': 5000, 'name': 'Alice'},
            {'age': 30, 'salary': 6000, 'name': 'Bob'},
            {'age': 35, 'salary': 7000, 'name': 'Charlie'},
        ]
    
    def test_z_score_scaling(self):
        """Test Z-score scaling for numeric columns"""
        result = ScaleNumeric.process(self.data, ['age', 'salary'])

        # Calculate mean and standard deviation for age and salary
        age_values = [row['age'] for row in self.data]
        salary_values = [row['salary'] for row in self.data]

        # Expected calculations for mean and std deviation
        age_mean = sum(age_values) / len(age_values)
        age_std_dev = math.sqrt(sum((x - age_mean) ** 2 for x in age_values) / len(age_values))

        salary_mean = sum(salary_values) / len(salary_values)
        salary_std_dev = math.sqrt(sum((x - salary_mean) ** 2 for x in salary_values) / len(salary_values))

        # Verify that the scaling has been done correctly (Z-score scaling)
        for i, row in enumerate(result):
            scaled_age = (row['age'] - age_mean) / age_std_dev if age_std_dev != 0 else 0
            scaled_salary = (row['salary'] - salary_mean) / salary_std_dev if salary_std_dev != 0 else 0

            self.assertAlmostEqual(row['age'], scaled_age, places=6)
            self.assertAlmostEqual(row['salary'], scaled_salary, places=6)

    def test_zero_standard_deviation(self):
        """Test scaling when standard deviation is zero (i.e., all values are the same)"""
        data_with_same_values = [
            {'age': 30, 'salary': 6000, 'name': 'Alice'},
            {'age': 30, 'salary': 6000, 'name': 'Bob'},
            {'age': 30, 'salary': 6000, 'name': 'Charlie'},
        ]

        result = ScaleNumeric.process(data_with_same_values, ['age', 'salary'])

        # Since all the values are the same, the Z-score normalization should result in 0 for all rows
        for row in result:
            self.assertEqual(row['age'], 0.0)
            self.assertEqual(row['salary'], 0.0)

    def test_scaling_single_column(self):
        """Test when only one numeric column is scaled"""
        data_with_single_column = [
            {'age': 25, 'salary': 5000, 'name': 'Alice'},
            {'age': 30, 'salary': 6000, 'name': 'Bob'},
            {'age': 35, 'salary': 7000, 'name': 'Charlie'},
        ]
        
        # Apply scaling on only the 'age' column
        result = ScaleNumeric.process(data_with_single_column, ['age'])

        # Calculate mean and std deviation for 'age'
        age_values = [row['age'] for row in data_with_single_column]
        age_mean = sum(age_values) / len(age_values)
        age_std_dev = math.sqrt(sum((x - age_mean) ** 2 for x in age_values) / len(age_values))

        # Verify that only the 'age' column is scaled, 'salary' remains unchanged
        for i, row in enumerate(result):
            scaled_age = (row['age'] - age_mean) / age_std_dev if age_std_dev != 0 else 0
            self.assertAlmostEqual(row['age'], scaled_age, places=6)
            self.assertEqual(row['salary'], data_with_single_column[i]['salary'])

if __name__ == '__main__':
    unittest.main()
