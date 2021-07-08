from timeit import default_timer as timer

import matplotlib.pyplot as plt
import sympy
import tensorflow as tf

from utils import *

script_start = timer()

np.random.seed(0)
tf.random.set_seed(0)

print("generating random qnn")
generated_circuit = generate_random_qnn(cirq.GridQubit.rect(1, 3), sympy.Symbol('theta'), 2)
print(generated_circuit)

n_qubits = [2 * i for i in range(1, 12)]  # Ranges studied in paper are between 2 and 24.
depth = 50  # Ranges studied in paper are between 50 and 500.
n_circuits = 200
theta_mean = []
theta_var = []

for n in n_qubits:
    # Generate the random circuits and observable for the given n.
    print("generating the random circuits and observable for n={}".format(n))
    start = timer()
    qubits = cirq.GridQubit.rect(1, n)
    symbol = sympy.Symbol('theta')
    circuits = [generate_random_qnn(qubits, symbol, depth) for _ in range(n_circuits)]
    op = cirq.Z(qubits[0]) * cirq.Z(qubits[1])
    mean, var = process_batch(circuits, symbol, op)
    theta_mean.append(mean)
    theta_var.append(var)
    print("elapsed time:", (timer()-start))

plt.plot(n_qubits, theta_mean, label="random init")
plt.title('Gradient Mean (200 random circuits)')
plt.xlabel('n_qubits')
plt.ylabel('$\\partial \\theta$ mean')
plt.legend()
plt.xticks(n_qubits)
plt.show()

plt.semilogy(n_qubits, theta_var, label="random init")
plt.title('Gradient Variance (200 random circuits)')
plt.xlabel('n_qubits')
plt.ylabel('$\\partial \\theta$ variance')
plt.legend()
plt.xticks(n_qubits)
plt.show()

print("generating identity qnn")
generate_identity_qnn(cirq.GridQubit.rect(1, 3), sympy.Symbol('theta'), 2, 2)

block_depth = 10
total_depth = 5

heuristic_theta_mean = []
heuristic_theta_var = []

for n in n_qubits:
    # Generate the identity block circuits and observable for the given n.
    print("generating the identity block and observable for n={}".format(n))
    start = timer()
    qubits = cirq.GridQubit.rect(1, n)
    symbol = sympy.Symbol('theta')
    circuits = [generate_identity_qnn(qubits, symbol, block_depth, total_depth) for _ in range(n_circuits)]
    op = cirq.Z(qubits[0]) * cirq.Z(qubits[1])
    mean, var = process_batch(circuits, symbol, op)
    heuristic_theta_mean.append(mean)
    heuristic_theta_var.append(var)
    print("elapsed time:", (timer()-start))

plt.plot(n_qubits, theta_mean, label="random init")
plt.plot(n_qubits, heuristic_theta_mean, label="identity init")
plt.title('Gradient Mean (200 random circuits)')
plt.xlabel('n_qubits')
plt.ylabel('$\\partial \\theta$ mean')
plt.legend()
plt.xticks(n_qubits)
plt.show()

plt.semilogy(n_qubits, theta_var, label="random init")
plt.semilogy(n_qubits, heuristic_theta_var, label="identity init")
plt.title('Gradient Variance (200 random circuits)')
plt.xlabel('n_qubits')
plt.ylabel('$\\partial \\theta$ variance')
plt.legend()
plt.xticks(n_qubits)
plt.show()

print("Script execution time: ", (timer()-script_start))
print("Done")
