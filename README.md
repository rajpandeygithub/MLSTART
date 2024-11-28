# MLStart: Simplified Machine Learning Pipeline

MLStart is a Python library designed for beginners and non-experts in machine learning. It automates essential steps like data preparation, model training, evaluation, and report generation. MLStart intelligently detects whether your dataset is suited for classification or regression tasks, allowing you to get started without needing to decide in advance. With MLStart, you can gain a baseline understanding of your dataset and choose the best-performing model with minimal effort.

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
- Your dataset must be a CSV file and placed in the `data/` folder.
- Provide the name of the target column (the column you want to predict).
- **For Classification**: The dataset should have balanced target classes (e.g., no highly imbalanced target classes).

This keeps the instructions clear and simple while specifying the balance requirement for classification tasks.

### Example Code

```python
from mlstart.pipeline import MLStartPipeline

# Initialize the pipeline with your dataset file and target column
pipeline = MLStartPipeline(file_name="your_dataset.csv", target_column="target_column_name")

# Run the pipeline to get the evaluation and recommendations
pipeline.evaluate_and_recommend_model()
```

After running the pipeline:
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
.
├── data/                 # Place your input CSV files here
├── report_output/        # Contains generated reports
├── core/                 # Core modules like data loading and task identification
├── processing/           # Preprocessing and data preparation modules
├── models/               # Model training and evaluation modules
├── reporting/            # Report generation module
├── requirements.txt      # Dependencies for the project
├── README.md             # Project documentation
```

---

## Limitations
- MLStart is designed for creating **baseline models**. It’s not intended for fine-tuning or advanced model customization.
- Currently, class imbalance is not automatically handled. Ensure your dataset is balanced before running the pipeline.

---

## Why Use MLStart?

If you're new to machine learning and want a quick, easy way to explore your data and find the best model to start with, MLStart is the perfect tool for you.
