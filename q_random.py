import cirq

# Create a circuit to generate a Bell State:
# sqrt(2) * ( |00> + |11> )
circuit = cirq.Circuit()
q0 = cirq.NamedQubit("qubit0")
circuit += cirq.H(q0)

# Initialize Simulator
s = cirq.Simulator()

print('Simulate the circuit:\n' + str(circuit))
results = s.simulate(circuit)
print(results, "\n")

# For sampling, we need to add a measurement at the end
circuit.append(cirq.measure(q0, key='result'))

print('Sample the circuit:')
print(circuit)
samples = s.run(circuit, repetitions=10)
print(samples)
# # Print a histogram of results
print(samples.histogram(key='result'))
