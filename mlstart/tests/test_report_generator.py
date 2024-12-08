import unittest
from mlstart.reporting.report_generator import ReportGenerator

class TestReportGenerator(unittest.TestCase):

    def setUp(self):
        """Set up example data for testing"""
        self.model_metrics = {
            "Model A": {"Accuracy": 0.85, "Precision": 0.80, "Recall": 0.75, "F1 Score": 0.77},
            "Model B": {"Accuracy": 0.88, "Precision": 0.85, "Recall": 0.78, "F1 Score": 0.81},
            "Model C": {"Accuracy": 0.82, "Precision": 0.79, "Recall": 0.74, "F1 Score": 0.76},
        }
        self.rankings = {
            "Model A": {"Accuracy": 2, "Precision": 3, "Recall": 3, "F1 Score": 3, "average_rank": 2.75},
            "Model B": {"Accuracy": 1, "Precision": 1, "Recall": 1, "F1 Score": 1, "average_rank": 1.0},
            "Model C": {"Accuracy": 3, "Precision": 2, "Recall": 2, "F1 Score": 2, "average_rank": 2.25},
        }
        self.best_model_name = "Model B"
        self.report_generator = ReportGenerator(task_type="classification")

    def test_generate_report(self):
        """Test if the report is generated correctly"""
        report = self.report_generator.generate_report(self.model_metrics, self.best_model_name, self.rankings)
        
        # Check that the report contains the expected sections
        self.assertIn("Model Performance Report (Classification Task)", report)
        self.assertIn("Summary Table (All Metrics at a Glance):", report)
        self.assertIn("Metric: Accuracy (Higher is Better)", report)
        self.assertIn("Best Model: Model B", report)

    def test_save_report(self):
        """Test if the report is saved to a file correctly"""
        report = self.report_generator.generate_report(self.model_metrics, self.best_model_name, self.rankings)
        file_path = "test_report.txt"
        
        # Save the report to a file
        self.report_generator.save_report(report, file_path=file_path)
        
        # Check if the file is created and contains the expected content
        with open(file_path, "r") as file:
            file_content = file.read()
            self.assertIn("Model Performance Report (Classification Task)", file_content)

if __name__ == '__main__':
    unittest.main()
