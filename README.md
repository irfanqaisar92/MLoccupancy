# MLoccupancy

# Occupancy Prediction with Machine Learning

This repository contains a Python script for predicting occupancy levels ("low" or "high") using multiple machine learning models, based on environmental and sensor data. The script processes CSV files, trains models with different weekday-based strategies, evaluates performance with metrics like accuracy, recall, precision, F1-score, and confusion matrix, and saves the results.

## Overview

The script:
- Loads time-series data from CSV files.
- Splits data into training and testing sets based on weekday strategies.
- Scales features for better model performance.
- Trains four models: Logistic Regression, Random Forest, Decision Tree, and Gradient Boosting (XGBoost).
- Evaluates and saves predictions and metrics.

## Prerequisites

- **Python 3.8+**: Ensure Python is installed.
- **Conda**: For environment management (optional but recommended).
- **Git**: To clone this repository.

## Setup

### 1. Clone the Repository
Clone this repo to your local machine:
```
git clone https://github.com/irfanqaisar92/MLoccupancy.git
cd occupancy-prediction
```

### 2. Create a Conda Environment
Set up a virtual environment with required libraries:

```
conda create --name ml_env python=3.8
conda activate ml_env
conda install pandas scikit-learn numpy
```
### 3. Prepare Your Data
Place your CSV files in a folder named Beehub_ML within the project directory. The script expects:

A datetime or timestamp column in "DD/MM/YYYY HH:MM" format (e.g., "16/03/2020 8:00").
Features like temperature, humidity, airflow, etc.
A target column named occ with values "low" or "high".
An optional index column (handled automatically for files like file3.csv).

#### Example structure:

```
occupancy-prediction/
├── Beehub_ML/
│   ├── file1.csv
│   ├── file2.csv
│   ├── file3.csv
├── ML.py
```
#### Sample CSV format:
```
timestamp,Dry Bulb Temperature,Absolute Humidity,Relative Humidity,lights,plug,occ
16/03/2020 8:00,28.15,21.33,77.52,0.41,0.995,low
16/03/2020 8:30,28.42,21.46,77.10,0.45,0.675,low
```
## Usage
### 1. Run the Script
Execute the script from the project directory:
```
python ML.py
```
### 2. Script Details
Here’s the full script (ML.py):

```
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(filepath):
    # Load the dataset, handling possible index column and setting datetime as index
    data = pd.read_csv(filepath, index_col=0 if filepath.endswith('file3.csv') else None)
    if 'timestamp' in data.columns:
        data = data.rename(columns={'timestamp': 'datetime'})
    data['datetime'] = pd.to_datetime(data['datetime'], format='%d/%m/%Y %H:%M')
    data.set_index('datetime', inplace=True)
    data['occ'] = data['occ'].map({'low': 0, 'high': 1})
    return data

def prepare_data(data, strategy):
    if strategy == 1:
        train_data = data[data.index.weekday < 4]  # Mon-Thu
        test_data = data[data.index.weekday == 4]  # Fri
    elif strategy == 2:
        train_data = data[data.index.weekday < 3]  # Mon-Wed
        test_data = data[data.index.weekday >= 3]  # Thu-Fri
    elif strategy == 3:
        train_data = data[data.index.weekday < 2]  # Mon-Tue
        test_data = data[data.index.weekday >= 2]  # Wed-Fri
    elif strategy == 4:
        train_data = data[data.index.weekday == 0]  # Mon
        test_data = data[data.index.weekday > 0]   # Tue-Fri

    X_train = train_data.drop(columns='occ')
    y_train = train_data['occ']
    X_test = test_data.drop(columns='occ')
    y_test = test_data['occ']

    # Preserve the index before scaling
    X_test_index = X_test.index

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test, X_test_index

def evaluate_metrics(y_true, y_pred):
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if cm.shape == (1, 1):  # If only one class exists in y_true
        return 0.0, 0.0, 0.0, 0.0, np.array([[0, 0], [0, 0]])
    tn = cm[0, 0] if cm.shape[0] > 1 else 0
    tp = cm[1, 1] if cm.shape[0] > 1 else 0
    fn = cm[1, 0] if cm.shape[0] > 1 else 0
    fp = cm[0, 1] if cm.shape[0] > 1 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return round(accuracy * 100, 2), round(recall * 100, 2), round(precision * 100, 2), round(f1 * 100, 2), cm

def train_and_evaluate(X_train, y_train, X_test, y_test, filename, strategy):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'XGBoost': GradientBoostingClassifier()
    }

    results = {}
    metrics = []

    print(f"\n--- Evaluation Results for {filename} (Strategy {strategy}) ---")

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy, recall, precision, f1, cm = evaluate_metrics(y_test, predictions)

        print(f'[{name}]')
        print(f'  Accuracy  : {accuracy:.2f}%')
        print(f'  Recall    : {recall:.2f}%')
        print(f'  Precision : {precision:.2f}%')
        print(f'  F1-Score  : {f1:.2f}%')
        print(f'  Confusion Matrix:\n{cm}\n')
        print("---------------------------------------------------")

        results[name] = predictions
        metrics.append({
            'File': filename,
            'Strategy': strategy,
            'Model': name,
            'Accuracy': accuracy,
            'Recall': recall,
            'Precision': precision,
            'F1': f1,
            'Confusion Matrix': cm.tolist()
        })

    return results, metrics

def main():
    data_directory = './Beehub_ML'
    result_directory = './Beehub_ML_Results'
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    strategies = {
        1: "Mon-Thur train, Fri test",
        2: "Mon-Wed train, Thu-Fri test",
        3: "Mon-Tue train, Wed-Fri test",
        4: "Mon train, Tue-Fri test"
    }

    all_metrics = []

    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(data_directory, filename)
            data = load_data(filepath)
            for strategy, description in strategies.items():
                X_train, y_train, X_test, y_test, X_test_index = prepare_data(data, strategy)
                if len(X_train) == 0 or len(X_test) == 0:
                    print(f"Skipping {filename} (Strategy {strategy}): No data for train or test split.")
                    continue
                results, metrics = train_and_evaluate(X_train, y_train, X_test, y_test, filename, strategy)
                all_metrics.extend(metrics)

                # Use the preserved index for the DataFrame
                combined_results = pd.DataFrame(index=X_test_index)
                combined_results['True Values'] = y_test.map({0: 'low', 1: 'high'})
                for model_name, predictions in results.items():
                    combined_results[f'{model_name} Predictions'] = predictions
                    combined_results[f'{model_name} Predictions'] = combined_results[f'{model_name} Predictions'].map({0: 'low', 1: 'high'})
                csv_filename = f"{result_directory}/{filename.replace('.csv', '')}_strategy_{strategy}_all_predictions.csv"
                combined_results.to_csv(csv_filename)
                print(f"All model predictions for strategy {strategy} ({description}) saved to {csv_filename}")

    metrics_df = pd.DataFrame(all_metrics)

    print("\n=== All Accuracy Results ===")
    for _, row in metrics_df.iterrows():
        print(f"File: {row['File']}, Strategy: {row['Strategy']}, Model: {row['Model']}, Accuracy: {row['Accuracy']:.2f}%")

    print("\n=== All Recall Results ===")
    for _, row in metrics_df.iterrows():
        print(f"File: {row['File']}, Strategy: {row['Strategy']}, Model: {row['Model']}, Recall: {row['Recall']:.2f}%")

    print("\n=== All Precision Results ===")
    for _, row in metrics_df.iterrows():
        print(f"File: {row['File']}, Strategy: {row['Strategy']}, Model: {row['Model']}, Precision: {row['Precision']:.2f}%")

    print("\n=== All F1-Score Results ===")
    for _, row in metrics_df.iterrows():
        print(f"File: {row['File']}, Strategy: {row['Strategy']}, Model: {row['Model']}, F1-Score: {row['F1']:.2f}%")

    print("\n=== All Confusion Matrix Results ===")
    for _, row in metrics_df.iterrows():
        cm = np.array(row['Confusion Matrix'])
        print(f"File: {row['File']}, Strategy: {row['Strategy']}, Model: {row['Model']}, Confusion Matrix:\n{cm}\n")

    metrics_df.to_csv(os.path.join(result_directory, 'all_metrics.csv'), index=False)
    print(f"\nAll metrics saved to {os.path.join(result_directory, 'all_metrics.csv')}")

if __name__ == "__main__":
    main()
```
### 3. Output
#### Console:
Displays metrics for each model, file, and strategy, followed by a summary of all results.
#### Files:
Results are saved in Beehub_ML_Results/:
Prediction CSVs (e.g., file1_strategy_1_all_predictions.csv) with true values and model predictions.
all_metrics.csv with aggregated metrics, including confusion matrices.

## Example console output:
```
--- Evaluation Results for file1.csv (Strategy 1) ---
[Logistic Regression]
  Accuracy  : 85.00%
  Recall    : 80.00%
  Precision : 88.89%
  F1-Score  : 84.21%
  Confusion Matrix:
[[9 1]
 [2 8]]
```
```
=== All Accuracy Results ===
File: file1.csv, Strategy: 1, Model: Logistic Regression, Accuracy: 85.00%
...
Interpreting Results
Accuracy: Percentage of correct predictions.
Recall: Ability to identify "high" occupancy (true positives / (true positives + false negatives)).
Precision: Accuracy of "high" predictions (true positives / (true positives + false positives)).
F1-Score: Harmonic mean of precision and recall.
Confusion Matrix: [[TN, FP], [FN, TP]] where:
TN: True Negatives ("low" correctly predicted).
FP: False Positives ("low" predicted as "high").
FN: False Negatives ("high" predicted as "low").
TP: True Positives ("high" correctly predicted).
```
## Customization
#### Change File Names:
Update load_data if your file naming differs (e.g., replace file3.csv with your actual filename for index handling).
#### Adjust Strategies:
Modify the strategies dictionary in main to change train/test splits.
#### Add Models:
Extend the models dictionary in train_and_evaluate with other scikit-learn models.

## Troubleshooting
#### ConvergenceWarning:
If Logistic Regression doesn’t converge, increase max_iter or try solver='liblinear':
```
'Logistic Regression': LogisticRegression(max_iter=5000, solver='liblinear'),
```
#### Missing Data:
Ensure your CSV files cover all weekdays to avoid skipped strategies.
#### Contributing
Feel free to fork this repo, make improvements, and submit a pull request!
