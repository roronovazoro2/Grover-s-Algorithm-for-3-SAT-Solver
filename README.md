# Grover's Algorithm for 3-SAT Solver

This implementation uses Grover's quantum algorithm to solve 3-SAT problems, demonstrating quantum speed-up in solving logic-based problems.

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The implementation provides a `Grover3SAT` class that can solve 3-SAT problems using Grover's algorithm. Here's a basic example:

```python
from grover_3sat import Grover3SAT

# Define your 3-SAT formula
# Example: (x1 OR ¬x2 OR x3) AND (¬x1 OR x2 OR ¬x3)
clauses = [
    [(0, False), (1, True), (2, False)],  # (x1 OR ¬x2 OR x3)
    [(0, True), (1, False), (2, True)]    # (¬x1 OR x2 OR ¬x3)
]

# Create and solve the problem
solver = Grover3SAT(clauses)
results = solver.solve()

# Print and visualize results
print("Measurement results:")
print(results)
solver.plot_results(results)
```

## How it Works

1. The algorithm takes a list of clauses, where each clause is a tuple of 3 literals.
2. Each literal is represented as (variable_index, is_negated).
3. The implementation:
   - Creates a quantum circuit with the necessary number of qubits
   - Implements the oracle that marks satisfying assignments
   - Applies Grover's diffusion operator
   - Measures the results to find the most probable satisfying assignments

## Features

- Supports arbitrary 3-SAT formulas
- Automatically determines optimal number of Grover iterations
- Visualizes results using histograms
- Uses Qiskit's quantum circuit simulator

## Requirements

- Python 3.7+
- Qiskit
- NumPy
- Matplotlib

## Note

This implementation uses Qiskit's quantum circuit simulator. For real quantum hardware execution, you would need to modify the code to use IBM Quantum backends. 