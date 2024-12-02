import unittest

# Import test modules
from mlstart.tests import test_data_loader
"""from mlstart.tests import test_task_identifier
from mlstart.tests import test_column_identifier
from mlstart.tests import test_model_trainer
from mlstart.tests import test_model_comparator
from mlstart.tests import test_report_generator"""

# Initialize test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

# Add tests to the test suite
suite.addTest(loader.loadTestsFromModule(test_data_loader))
"""suite.addTest(loader.loadTestsFromModule(test_task_identifier))
suite.addTest(loader.loadTestsFromModule(test_column_identifier))
suite.addTest(loader.loadTestsFromModule(test_model_trainer))
suite.addTest(loader.loadTestsFromModule(test_model_comparator))
suite.addTest(loader.loadTestsFromModule(test_report_generator))"""

# Initialize a test runner and run the test suite
runner = unittest.TextTestRunner(verbosity=2)
result = runner.run(suite)
