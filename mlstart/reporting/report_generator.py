class ReportGenerator:
    """
    A class to generate and display reports comparing model performance.
    """

    def __init__(self, task_type):
        """
        Initializes the ReportGenerator class.

        :param task_type: A string indicating the type of task ('classification' or 'regression').
        """
        self.task_type = task_type

    def generate_report(self, model_metrics, best_model_name, rankings):
        """
        Generates a detailed report with a summary table and detailed metric rankings.

        :param model_metrics: Dictionary containing model names and their evaluation metrics.
        :param best_model_name: Name of the best-performing model.
        :param rankings: Rankings of models across all metrics.
        :return: A formatted detailed report string.
        """
        report_lines = []
        report_lines.append(f"Model Performance Report ({self.task_type.capitalize()} Task)")
        report_lines.append("=" * 50)

        # Summary Table Section
        report_lines.append("\nSummary Table (All Metrics at a Glance):")
        header = ["Model"] + list(next(iter(model_metrics.values())).keys()) + ["Average Rank"]
        header_line = "{:<25}".format(header[0]) + "".join(f"{key:<15}" for key in header[1:])
        separator_line = "-" * len(header_line)
        report_lines.append(header_line)
        report_lines.append(separator_line)

        for model_name, metrics in model_metrics.items():
            row = "{:<25}".format(model_name)
            for key in metrics.keys():
                metric_value = metrics[key]
                if abs(metric_value) >= 10000 or abs(metric_value) < 0.001:  # Large/small values
                    row += f"{metric_value:<15.3e}"
                else:
                    row += f"{metric_value:<15.4f}"
            average_rank = rankings[model_name]["average_rank"]
            row += f"{average_rank:<15.2f}"
            report_lines.append(row)

        # Metric-specific sections
        for metric in next(iter(model_metrics.values())).keys():
            # Add a descriptive title for the section
            description = "Higher is Better" if metric not in ["MAE", "MSE", "RMSE"] else "Lower is Better"
            explanation = {
                "Accuracy": "Measures the proportion of correct predictions.",
                "Precision": "Measures how many selected items are relevant.",
                "Recall": "Measures how many relevant items are selected.",
                "F1 Score": "Harmonic mean of precision and recall.",
                "MAE": "Mean Absolute Error: Average of absolute errors.",
                "MSE": "Mean Squared Error: Average of squared errors.",
                "RMSE": "Root Mean Squared Error: Square root of MSE."
            }.get(metric, "")
            report_lines.append(f"\nMetric: {metric} ({description})")
            report_lines.append(f"{explanation}")
            report_lines.append("-" * 50)

            # Table header
            header = "{:<25} {:<15} {:<5}".format("Model", metric, "Rank")
            report_lines.append(header)
            report_lines.append("-" * len(header))

            # Populate table rows
            for model_name, metrics in model_metrics.items():
                value = metrics[metric]
                rank = rankings[model_name][metric]
                if abs(value) >= 10000 or abs(value) < 0.001:
                    value_str = f"{value:.3e}"  # Scientific notation for large/small values
                else:
                    value_str = f"{value:.4f}"  # Fixed-point notation for normal values
                row = "{:<25} {:<15} {:<5}".format(model_name, value_str, rank)
                report_lines.append(row)

        # Overall Rankings Section
        report_lines.append("\nOverall Rankings")
        report_lines.append("=" * 50)
        report_lines.append("{:<25} {:<15}".format("Model", "Average Rank"))
        report_lines.append("-" * 50)
        for model_name in rankings.keys():
            avg_rank = rankings[model_name]["average_rank"]
            best_indicator = " *Best*" if model_name == best_model_name else ""
            row = "{:<25} {:<15.2f}{}".format(model_name, avg_rank, best_indicator)
            report_lines.append(row)

        # Recommendation Section
        report_lines.append("=" * 50)
        report_lines.append(f"\nBest Model: {best_model_name}")
        report_lines.append(
            f"Based on the evaluation, {best_model_name} is the top-performing model for your dataset.\n"
            f"We recommend starting with this model for your task as it has the best overall performance."
        )

        return "\n".join(report_lines)



    def save_report(self, report, file_path="report_output/model_report.txt"):
        """
        Saves the generated report to a text file.

        :param report: The report string to be saved.
        :type report: str
        :param file_path: The file path (including the file name) where the report will be saved.
                         If the directory does not exist, it will be created automatically.
        :type file_path: str
        """
        with open(file_path, "w") as file:
            file.write(report)
        print(f"The final Report has been saved at {file_path} location")