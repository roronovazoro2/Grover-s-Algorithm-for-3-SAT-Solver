from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import MCZGate
import numpy as np
import matplotlib.pyplot as plt

class Grover3SAT:
    def __init__(self, clauses):
        """
        Initialize the Grover 3-SAT solver.
        
        Args:
            clauses (list): List of clauses, where each clause is a tuple of 3 literals.
                           Each literal is represented as (variable_index, is_negated)
                           e.g., [(0, False), (1, True), (2, False)] represents (x1 OR ¬x2 OR x3)
        """
        self.clauses = clauses
        self.num_variables = max(max(abs(lit[0]) for lit in clause) for clause in clauses)
        
        # Create quantum registers
        self.qr = QuantumRegister(self.num_variables, 'q')
        self.ancilla = QuantumRegister(1, 'ancilla')
        self.cr = ClassicalRegister(self.num_variables, 'c')
        
        # Create the main circuit
        self.circuit = QuantumCircuit(self.qr, self.ancilla, self.cr)
        
    def _create_oracle(self):
        """Create the oracle circuit that marks satisfying assignments."""
        oracle = QuantumCircuit(self.qr, self.ancilla)
        
        # Initialize ancilla in |-> state
        oracle.h(self.ancilla[0])
        oracle.z(self.ancilla[0])
        
        # For each clause, apply controlled operations
        for clause in self.clauses:
            # Create a temporary circuit for this clause
            clause_circuit = QuantumCircuit(self.qr, self.ancilla)
            
            # Apply X gates to negated literals
            for var_idx, is_negated in clause:
                if is_negated:
                    clause_circuit.x(self.qr[var_idx])
            
            # Apply multi-controlled Z gate
            control_qubits = [self.qr[var_idx] for var_idx, _ in clause]
            mcz = MCZGate(len(control_qubits))
            clause_circuit.append(mcz, control_qubits + [self.ancilla[0]])
            
            # Uncompute X gates
            for var_idx, is_negated in clause:
                if is_negated:
                    clause_circuit.x(self.qr[var_idx])
            
            # Append this clause's circuit to the oracle
            oracle.compose(clause_circuit, inplace=True)
        
        return oracle
    
    def _create_diffuser(self):
        """Create the Grover diffusion operator."""
        diffuser = QuantumCircuit(self.qr)
        
        # Apply H gates to all qubits
        diffuser.h(self.qr)
        
        # Apply X gates to all qubits
        diffuser.x(self.qr)
        
        # Apply multi-controlled Z gate
        diffuser.h(self.qr[-1])
        mcz = MCZGate(len(self.qr)-1)
        diffuser.append(mcz, list(self.qr))
        diffuser.h(self.qr[-1])
        
        # Uncompute X gates
        diffuser.x(self.qr)
        
        # Uncompute H gates
        diffuser.h(self.qr)
        
        return diffuser
    
    def solve(self, num_iterations=None):
        """
        Solve the 3-SAT problem using Grover's algorithm.
        
        Args:
            num_iterations (int, optional): Number of Grover iterations.
                                          If None, will use optimal number based on problem size.
        
        Returns:
            dict: Measurement results showing the most probable satisfying assignments
        """
        if num_iterations is None:
            # Optimal number of iterations is approximately sqrt(2^n/num_solutions)
            # For simplicity, we'll use sqrt(2^n) as an approximation
            num_iterations = int(np.sqrt(2**self.num_variables))
        
        # Create the oracle and diffuser
        oracle = self._create_oracle()
        diffuser = self._create_diffuser()
        
        # Initialize the circuit
        self.circuit.h(self.qr)  # Create superposition
        
        # Apply Grover iterations
        for _ in range(num_iterations):
            self.circuit.compose(oracle, inplace=True)
            self.circuit.compose(diffuser, inplace=True)
        
        # Measure the qubits
        self.circuit.measure(self.qr, self.cr)
        
        # Run the circuit
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(self.circuit, simulator, shots=1024)
        result = job.result()
        counts = result.get_counts()
        
        return counts
    
    def plot_results(self, counts):
        """Plot the measurement results."""
        plot_histogram(counts)
        plt.show()

# Example usage
if __name__ == "__main__":
    # Example 3-SAT formula: (x1 OR ¬x2 OR x3) AND (¬x1 OR x2 OR ¬x3)
    clauses = [
        [(0, False), (1, True), (2, False)],  # (x1 OR ¬x2 OR x3)
        [(0, True), (1, False), (2, True)]    # (¬x1 OR x2 OR ¬x3)
    ]
    
    # Create and solve the 3-SAT problem
    solver = Grover3SAT(clauses)
    results = solver.solve()
    
    # Print and plot results
    print("Measurement results:")
    print(results)
    solver.plot_results(results) 