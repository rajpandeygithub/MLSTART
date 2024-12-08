import unittest

# Import test modules

from mlstart.tests import test_column_identifier
from mlstart.tests import test_data_loader
from mlstart.tests import test_datahandler
from mlstart.tests import test_encode_categorical
from mlstart.tests import test_handle_outlier
from mlstart.tests import test_model_comparator
from mlstart.tests import test_model_evaluator
from mlstart.tests import test_model_trainer
from mlstart.tests import test_normalize_missing_value
from mlstart.tests import test_remove_duplicates
from mlstart.tests import test_remove_invalid_columns
from mlstart.tests import test_report_generator
from mlstart.tests import test_scale_numeric
from mlstart.tests import test_task_identifier

# Initialize test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Add tests to the test suite

suite.addTest(loader.loadTestsFromModule(test_column_identifier))
suite.addTest(loader.loadTestsFromModule(test_data_loader))
suite.addTest(loader.loadTestsFromModule(test_datahandler))
suite.addTest(loader.loadTestsFromModule(test_encode_categorical))
suite.addTest(loader.loadTestsFromModule(test_handle_outlier))
suite.addTest(loader.loadTestsFromModule(test_model_comparator))
suite.addTest(loader.loadTestsFromModule(test_model_evaluator))
suite.addTest(loader.loadTestsFromModule(test_model_trainer))
suite.addTest(loader.loadTestsFromModule(test_normalize_missing_value))
suite.addTest(loader.loadTestsFromModule(test_remove_duplicates))
suite.addTest(loader.loadTestsFromModule(test_remove_invalid_columns))
suite.addTest(loader.loadTestsFromModule(test_report_generator))
suite.addTest(loader.loadTestsFromModule(test_scale_numeric))
suite.addTest(loader.loadTestsFromModule(test_task_identifier))

# Initialize a test runner and run the test suite
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
