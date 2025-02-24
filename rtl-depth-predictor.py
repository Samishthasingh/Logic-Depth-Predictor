import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pyverilog.vparser.parser as parser
from pyverilog.dataflow.dataflow_analyzer import VerilogDataflowAnalyzer
from pyverilog.dataflow.optimizer import VerilogDataflowOptimizer


class RTLDepthPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False

    def extract_features(self, rtl_file, signal_name):
        """Extract features from RTL file for a given signal"""
        # Parse the Verilog file
        ast = parser.parse([rtl_file])

        # Initialize the dataflow analyzer
        analyzer = VerilogDataflowAnalyzer(ast)
        analyzer.generate()

        # Get the optimized dataflow graph
        optimizer = VerilogDataflowOptimizer(analyzer.getBinddict())
        optimizer.optimize()

        # Extract features for the target signal
        features = {}

        # Get signal definition
        signal_def = None
        for bind in analyzer.getBinddict().values():
            for bk, bv in bind.items():
                if bk == signal_name:
                    signal_def = bv
                    break
            if signal_def:
                break

        if not signal_def:
            raise ValueError(f"Signal {signal_name} not found in RTL")

        # Feature 1: Fanin count - number of signals that directly feed into our target signal
        fanin_signals = self._get_fanin_signals(signal_def, analyzer)
        features['fanin_count'] = len(fanin_signals)

        # Feature 2: Operator complexity - count of operations in signal definition
        # Different operators contribute differently to combinational depth
        code = str(signal_def)
        features['and_count'] = code.count('&')
        features['or_count'] = code.count('|')
        features['xor_count'] = code.count('^')
        features['not_count'] = code.count('~')
        features['add_count'] = code.count('+')
        features['sub_count'] = code.count('-')
        features['mul_count'] = code.count('*')
        features['div_count'] = code.count('/')
        features['comparator_count'] = code.count('>') + code.count('<') + code.count('==') + code.count('!=')

        # Feature 3: Expression complexity - count of parentheses as a proxy for expression complexity
        features['paren_count'] = code.count('(')

        # Feature 4: Conditional complexity - count of conditional operators
        features['conditional_count'] = code.count('?')

        # Feature 5: Statement depth - estimating the depth of the statement
        features['statement_depth'] = self._estimate_statement_depth(code)

        # Feature 6: Line count of the signal definition - longer definitions likely have more complexity
        features['def_line_count'] = len(code.split('\n'))

        # Feature 7: Recursive fanin depth - how deep the fanin tree goes
        features['fanin_depth'] = self._calculate_fanin_depth(signal_name, analyzer, optimizer)

        # Feature 8: Module nesting level - signals in deeply nested modules may have more complexity
        features['module_nesting'] = self._get_module_nesting(signal_name, ast)

        # Feature 9: Signal width - wider signals may have more complex logic
        features['signal_width'] = self._get_signal_width(signal_name, analyzer)

        # Feature 10: Number of conditionals affecting the signal
        features['if_count'] = self._count_conditionals(signal_name, analyzer)

        return features

    def _get_fanin_signals(self, signal_def, analyzer):
        """Get all signals that feed into the target signal"""
        # This is a simplified implementation - in a real system, you would trace the graph
        code = str(signal_def)
        # Use regex to find all signal names in the definition
        signal_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        signals = re.findall(signal_pattern, code)
        # Filter out keywords and the signal itself
        keywords = ['if', 'else', 'case', 'default', 'begin', 'end']
        signals = [s for s in signals if s not in keywords and s != signal_name]
        return set(signals)

    def _estimate_statement_depth(self, code):
        """Estimate the logical depth of a statement based on operators and nesting"""
        # Count nested parentheses as a proxy for expression depth
        max_depth = 0
        current_depth = 0
        for char in code:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        return max_depth

    def _calculate_fanin_depth(self, signal_name, analyzer, optimizer, visited=None):
        """Calculate the maximum depth of fanin signals recursively"""
        if visited is None:
            visited = set()

        if signal_name in visited:
            return 0  # Avoid cycles

        visited.add(signal_name)

        # Get the signal definition
        signal_def = None
        for bind in analyzer.getBinddict().values():
            for bk, bv in bind.items():
                if bk == signal_name:
                    signal_def = bv
                    break
            if signal_def:
                break

        if not signal_def:
            return 0  # Leaf node or signal not found

        # Get fanin signals
        fanin_signals = self._get_fanin_signals(signal_def, analyzer)

        if not fanin_signals:
            return 1  # Base case: only this signal

        # Recursively find the maximum depth of fanin signals
        max_fanin_depth = 0
        for fanin in fanin_signals:
            depth = self._calculate_fanin_depth(fanin, analyzer, optimizer, visited.copy())
            max_fanin_depth = max(max_fanin_depth, depth)

        return max_fanin_depth + 1  # Add 1 for this level

    def _get_module_nesting(self, signal_name, ast):
        """Determine the module nesting level of a signal"""
        # Simplified implementation - in a real system, traverse the AST to find module nesting
        # Return a default value for now
        return 1

    def _get_signal_width(self, signal_name, analyzer):
        """Determine the bit width of a signal"""
        # Simplified implementation - would need to parse the signal definition
        # Return a default value for now
        return 32

    def _count_conditionals(self, signal_name, analyzer):
        """Count the number of if/case statements affecting this signal"""
        # Simplified implementation
        # In a real implementation, traverse the AST to count conditionals
        return 0

    def prepare_dataset(self, rtl_files, signal_names, actual_depths):
        """Prepare dataset from multiple RTL files and signals with known depths"""
        features_list = []

        for i, (rtl_file, signal_name) in enumerate(zip(rtl_files, signal_names)):
            try:
                features = self.extract_features(rtl_file, signal_name)
                features_list.append(features)
            except Exception as e:
                print(f"Error extracting features for {signal_name} in {rtl_file}: {e}")

        # Convert to DataFrame
        X = pd.DataFrame(features_list)
        y = np.array(actual_depths)

        return X, y

    def train(self, X, y):
        """Train the model on the prepared dataset"""
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model.fit(X_train_scaled, y_train)
        self.trained = True

        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Model Evaluation:")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"R² Score: {r2:.4f}")

        # Return test data for further analysis
        return X_test, y_test, y_pred

    def predict(self, rtl_file, signal_name):
        """Predict the combinational depth of a signal in an RTL file"""
        if not self.trained:
            raise ValueError("Model has not been trained yet")

        # Extract features for the signal
        features = self.extract_features(rtl_file, signal_name)

        # Convert to DataFrame and scale
        X = pd.DataFrame([features])
        X_scaled = self.scaler.transform(X)

        # Predict using the model
        predicted_depth = self.model.predict(X_scaled)[0]

        return predicted_depth

    def feature_importance(self):
        """Get feature importance from the trained model"""
        if not self.trained:
            raise ValueError("Model has not been trained yet")

        # Get feature importance from the model
        importances = self.model.feature_importances_
        feature_names = self.model.feature_names_in_

        # Create a DataFrame for easier viewing
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        return importance_df


# Example usage
if __name__ == "__main__":
    # This is a mock example - in a real scenario, you would have actual RTL files and depth data

    # Initialize the predictor
    predictor = RTLDepthPredictor()

    # Mock data for demonstration - in reality, this would come from synthesis reports
    rtl_files = ["module1.v", "module2.v", "module3.v"]  # Path to RTL files
    signal_names = ["signal_a", "signal_b", "signal_c"]  # Signals to analyze
    actual_depths = [10, 15, 5]  # Known combinational depths from synthesis

    print("Preparing dataset...")
    # In a real implementation, uncomment the following:
    # X, y = predictor.prepare_dataset(rtl_files, signal_names, actual_depths)
    # X_test, y_test, y_pred = predictor.train(X, y)

    # Analyze feature importance
    # importance_df = predictor.feature_importance()
    # print("Feature Importance:")
    # print(importance_df)

    # Make a prediction for a new signal
    # new_rtl = "new_module.v"
    # new_signal = "critical_path_signal"
    # predicted_depth = predictor.predict(new_rtl, new_signal)
    # print(f"Predicted combinational depth for {new_signal}: {predicted_depth:.2f}")

    print("Note: This is a mock implementation. To use this code with real data:")
    print("1. Prepare a dataset of RTL files with known combinational depths from synthesis reports")
    print("2. Train the model using this dataset")
    print("3. Use the trained model to predict depths for new signals")