from mlstart.core.data_loader import DataLoader
from mlstart.core.task_identifier import TaskIdentifier
from mlstart.core.column_identifier import ColumnIdentifier
from mlstart.processing.preprocessor_pipeline import PreprocessorPipeline
from mlstart.processing.datahandler import DataHandler
from mlstart.models.model_trainer import ModelTrainer
from mlstart.models.model_comparison import ModelComparator
from mlstart.reporting.report_generator import ReportGenerator
import os


class MLStartPipeline:
    """
    Main user-facing class for the MLStart library. Handles the entire ML pipeline
    from data loading to report generation.
    """

    def __init__(self, file_name, target_column):
        """
        Initializes the MLStartPipeline with the dataset file and target column.

        :param file_name: Name of the dataset file (must be in the 'data' folder).
        :type file_name: str
        :param target_column: Name of the target column in the dataset.
        :type target_column: str
        """
        self.file_name = os.path.splitext(file_name)[0] + ".txt"  # Assuming all datasets are in the 'data' folder
        self.file_path = f"data/{file_name}"
        self.target_column = target_column
        self.task_type = None
        self.data = None
        self.numeric_columns = []
        self.categorical_columns = []

    def evaluate_and_recommend_model(self):
        """
        Executes the entire MLStart pipeline: data loading, preprocessing, model training, 
        evaluation, and report generation.

        Steps:
            - Loads the dataset from the specified file path.
            - Determines whether the task is 'classification' or 'regression' based on the target column.
            - Identifies numeric and categorical columns in the dataset.
            - Prepares the data for modeling by handling missing values, encoding categorical variables,
              scaling numeric variables, and removing outliers (if applicable).
            - Splits the dataset into training and testing sets.
            - Trains multiple baseline models based on the task type.
            - Evaluates and ranks the trained models to identify the best-performing one.
            - Generates a detailed performance report and saves it as a text file.

        :raises Exception: If any step in the pipeline fails, an error message will be printed.
        """
        try:
            # Step 1: Load data
            #print("Loading data...")
            data_loader = DataLoader(self.file_path)
            headers, self.data = data_loader.load_data()

            # Step 2: Determine task type
            task_identifier = TaskIdentifier(self.data, headers, self.target_column)
            self.task_type = task_identifier.determine_task_type()

            # Step 3: Identify column types
            column_identifier = ColumnIdentifier(self.data)
            self.numeric_columns, self.categorical_columns = column_identifier.identify_column_types()

            # Step 4: Preprocess the data
            preprocessor = PreprocessorPipeline(self.task_type)
            preprocessed_data = preprocessor.run(self.data, self.numeric_columns, self.categorical_columns)

            # Step 5: Split the data
            data_handler = DataHandler(preprocessed_data, self.target_column, self.task_type)
            X_train, X_test, y_train, y_test = data_handler.train_test_splitting()

            # Handle case where training data contains only one class
            if len(set(y_train)) == 1:
                print(f"Training data contains only one class: {set(y_train)}. Classification requires at least two classes.")
                return

            # Step 6: Training models
            model_trainer = ModelTrainer(self.task_type)
            model_trainer.initialize_models()
            trained_models = model_trainer.train_models(X_train, y_train)

            # Step 7: Comparing models
            comparator = ModelComparator(self.task_type)
            best_model_name, best_model, model_metrics, total_ranks = comparator.compare_models(
                trained_models, X_test, y_test
            )

            # Step 8: Generating report
            report_generator = ReportGenerator(self.task_type)
            report = report_generator.generate_report(model_metrics, best_model_name, total_ranks)

            # Print the report
            print(report)

            # Save the report as a text file
            report_generator.save_report(report,f"report_output/model_report_{self.task_type}_{self.file_name}")

        except Exception as e:
            print(f"An error occurred: {e}")