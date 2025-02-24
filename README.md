# RTL Combinational Depth Predictor

## Project Overview
This project aims to predict the combinational logic depth of signals in behavioral RTL without running a full synthesis. This can significantly speed up the timing analysis process and help identify potential timing violations early in the design flow.

## Problem Statement
Timing analysis is a crucial step in the design of any complex IP/SoC. However, timing analysis reports are generated after synthesis is complete, which is a very time-consuming process. This leads to overall delays in project execution time as timing violations can require architectural refactoring.

Creating an AI algorithm to predict combinational logic depth of signals in behavioral RTL can greatly speed up this process.

## Solution Approach
The solution uses a machine learning approach to predict the combinational depth of signals based on features extracted from the RTL code. The approach consists of the following steps:

1. **Feature Engineering**: Extract relevant features from RTL code that can indicate combinational complexity
2. **Model Training**: Train a Random Forest Regressor on a dataset of signals with known depths
3. **Prediction**: Use the trained model to predict depths for new signals

## Features Extracted
The model uses the following features to predict combinational depth:

1. Fanin count (number of signals feeding into the target signal)
2. Operator complexity (counts of different operators: AND, OR, XOR, etc.)
3. Expression complexity (count of parentheses as a proxy for nesting)
4. Conditional complexity (count of conditional operators)
5. Statement depth (estimated depth of the statement)
6. Definition line count (length of the signal definition)
7. Recursive fanin depth (how deep the fanin tree goes)
8. Module nesting level (signals in deeply nested modules may have more complexity)
9. Signal width (wider signals may have more complex logic)
10. Number of conditionals affecting the signal

## Requirements
- Python 3.7+
- PyVerilog
- scikit-learn
- pandas
- numpy

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/Logic-Depth-Predictor.git
cd Logic-Depth-Predictor

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Preparing the Dataset
You need to prepare a dataset of RTL files and their corresponding combinational depths obtained from synthesis reports.

```python
from rtl_depth_predictor import RTLDepthPredictor

# Initialize the predictor
predictor = RTLDepthPredictor()

# Prepare your dataset
rtl_files = ["path/to/module1.v", "path/to/module2.v", ...]
signal_names = ["signal_a", "signal_b", ...]
actual_depths = [10, 15, ...]  # From synthesis reports

# Prepare the dataset
X, y = predictor.prepare_dataset(rtl_files, signal_names, actual_depths)
```

### Training the Model
```python
# Train the model
X_test, y_test, y_pred = predictor.train(X, y)

# Analyze feature importance
importance_df = predictor.feature_importance()
print(importance_df)
```

### Making Predictions
```python
# Predict depth for a new signal
new_rtl = "path/to/new_module.v"
new_signal = "critical_path_signal"
predicted_depth = predictor.predict(new_rtl, new_signal)
print(f"Predicted combinational depth: {predicted_depth:.2f}")
```

## Evaluation
The model is evaluated using the following metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

A lower MAE and RMSE, and a higher R² score indicate better prediction accuracy.

## Limitations and Future Work
- The current implementation uses PyVerilog for RTL parsing, which may have limitations with certain RTL constructs.
- The feature set could be expanded to include more sophisticated metrics such as control flow complexity.
- More advanced models such as neural networks could be explored for improved accuracy.
- Support for other HDLs like VHDL or SystemVerilog could be added.