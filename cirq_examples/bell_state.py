import cirq

# Create a circuit to generate a Bell State:
# sqrt(2) * ( |00> + |11> )
bell_circuit = cirq.Circuit()
q0, q1 = cirq.LineQubit.range(2)
bell_circuit.append(cirq.H(q0))
bell_circuit.append(cirq.CNOT(q0,q1))

# Initialize Simulator
s = cirq.Simulator()

print('Simulate the circuit:')
results = s.simulate(bell_circuit)
print(results)
print()

# For sampling, we need to add a measurement at the end
bell_circuit.append(cirq.measure(q0, q1, key='result'))

print('Sample the circuit:')
samples = s.run(bell_circuit, repetitions=1000)
# Print a histogram of results
print(samples.histogram(key='result'))
