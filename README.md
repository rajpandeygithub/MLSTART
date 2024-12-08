# MLStart: Simplified Machine Learning Pipeline

MLStart is a Python toolkit designed to simplify machine learning workflows for beginners and non-experts. It automates critical steps such as data preparation, model training, evaluation, and report generation, making it accessible for those with limited ML experience.

With MLStart, users only need to provide a dataset (CSV file) and specify the target column they wish to predict. The toolkit intelligently detects whether the dataset is suited for classification or regression tasks, eliminating the need for prior decision-making. It preprocesses the data, trains multiple baseline models, and evaluates their performance.

The output is a comprehensive performance report that highlights how the dataset performs across various models, ranks the models using key metrics, and recommends the best baseline model to start with. This allows users to quickly identify a suitable model, saving time and effort while gaining valuable insights into their dataset. MLStart is an ideal starting point for exploring machine learning models and understanding dataset behavior, especially for those new to the field.

---

## Key Features

1. **Ease of Use**: Automates the entire machine learning pipeline with minimal user input.
2. **Task Identification**: Automatically determines whether the problem is a classification or regression task.
3. **Preprocessing**: Automatically handles common issues with input data, such as missing values, inconsistent data types, invalid entries, encoding categorical data, and scaling numeric features, ensuring the dataset is prepared for modeling.
4. **Model Selection**: Trains multiple baseline models and evaluates their performance.
5. **Performance Report**: Generates a detailed report comparing model performance and recommends the best model.
6. **No Deep ML Knowledge Needed**: Designed for users new to machine learning.

---

## How It Works

### Steps Performed by MLStart:
1. **Load Dataset**: Reads your dataset from a CSV file.
2. **Identify Task**: Determines if the target variable is for classification or regression.
3. **Preprocess Data**: Cleans and prepares the dataset for modeling.
4. **Train Models**: Trains baseline models suitable for your task.
5. **Evaluate Models**: Compares model performance using metrics like accuracy (for classification) or MAE (for regression).
6. **Generate Report**: Provides a clear and detailed report with recommendations for the best model.

---

## Installation

MLStart requires Python 3.7 or above. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Input Requirements
- **File Format**: Your dataset must be a **CSV file** and placed in the `data/` folder.
- **Target Column**: Provide the name of the target column (the column you want to predict).
- **Numeric Columns**: Should only contain numeric values (e.g., integers, floats). Any invalid or non-numeric entries will be handled during preprocessing.
- **Categorical Columns**: Should represent categories as text or integers. For example, "Male/Female" or "1/0". Columns with invalid values (e.g., symbols like `-`, empty strings) will be treated as missing data and handled appropriately.
- **For Classification**: The target column should have **balanced classes** (e.g., not highly imbalanced target classes, like 95% of one class and 5% of another).
- **For Regression**: The target column should have numeric values.
---

#### How to Use MLStart (Example Code) 

1. **Prepare Your Dataset**:
   - Place your dataset (CSV file) in the `data/` folder.
   - Ensure the dataset meets the input requirements mentioned above.

2. **Create and Run a Python File**:
   - Open a text editor or IDE and create a new Python file (e.g., `run_mlstart.py`).
   - Add the following code:

     ```python
     from mlstart.run_pipeline import MLStartPipeline

     # Initialize the pipeline with your dataset file and target column
     pipeline = MLStartPipeline(file_name="your_dataset.csv", target_column="target_column_name")

     # Run the pipeline to get the evaluation and recommendations
     pipeline.evaluate_and_recommend_model()
     ```

   - Save the file in the same directory where the `data/` folder is located.

3. **Run the Python File**:
   - Open a terminal or command prompt.
   - Navigate to the folder containing your Python file.
   - Execute the file with the following command:

     ```bash
     python run_mlstart.py
     ```

4. **View the Results**:
   - A performance report will be displayed in the terminal.
   - The report will also be saved as a text file in the `report_output/` folder.

---

## Output Example

Here’s what the report includes:
- **Summary Table**: Metrics for all models at a glance.
- **Detailed Metrics**: Rankings for each metric like accuracy, precision, MAE, etc.
- **Overall Rankings**: The best-performing model based on combined rankings.
- **Recommendation**: A clear suggestion for which model to start with.

---

## Folder Structure

```plaintext
MLStart/
├── mlstart/              # All the code goes inside this main module folder
│   ├── core/             # Core modules like data loading and task identification
│   ├── processing/       # Preprocessing and data preparation modules
│   ├── models/           # Model training and evaluation modules
│   ├── reporting/        # Report generation module
│   ├── tests/            # Test cases for modules and functionality
│   ├── __init__.py       # Makes 'mlstart' a package
│   ├── run_pipeline.py   # Main pipeline orchestration script
├── data/                 # Input CSV files
├── report_output/        # Generated reports
├── example_usage.py      # Example usage of the library
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore file
├── run_tests.py          # Script to run all unit tests
```
---
## Testing (Only for developers)

MLStart includes a comprehensive suite of unit tests to validate its core functionalities. Developers can use these tests to verify changes or additions to the codebase.
You can run the tests using the run_tests.py script or directly using the unittest framework:

### Using `run_tests.py`
```bash
python run_tests.py
```

### Using `unittest`
```bash
python -m unittest discover -s mlstart/tests
```

This will execute all test cases and display a summary of the results.

#### Adding New Tests
If you are a developer and make changes to the codebase or add new features, ensure they are tested. Add test cases to the appropriate test modules in the `mlstart/tests/` folder.


## Limitations
- **Baseline Model Recommendation**: MLStart is designed to recommend an initial model for your machine learning task. It does not perform model fine-tuning, hyperparameter optimization, or advanced customization.
- **Data Preprocessing Scope**: While MLStart automates several preprocessing steps (e.g., handling missing values, normalizing data, and encoding categorical variables), it assumes that the dataset meets basic quality standards, such as meaningful features and balanced classes for classification tasks.
- **Limited Customization**: The package focuses on simplicity and automation, making it less suitable for scenarios that require extensive customization or integration with advanced machine learning pipelines.

---

## Why Use MLStart?

If you're new to machine learning and want a quick, easy way to explore your data and find the best model to start with, MLStart is the perfect tool for you.
