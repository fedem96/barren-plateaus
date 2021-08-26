# Barren plateaus

We explore the problem of barren plateaus [\[1\]](#McClean) in **Quantum Neural Networks**: a certain large family of random quantum circuits have gradients that vanish almost everywhere.

Also, we explore the identity heuristic [\[2\]](#Grant) initialization strategy as possible solution to overcome this problem.

This project follows the *TensorFlow Quantum* tutorial *Barren plateaus* [\[3\]](#TFQ).

## Requirements

+ tensorflow 2.1.0
+ tensorflow-quantum 0.3.1
 
 Install them with:
 
```sh
$ pip install -r requirements.txt
```

A quantum hardware is not required. The circuits are simulated using the *Cirq* library [\[4\]](#Cirq).

## Experiments

Run the experiments:

```sh
$ python3 barren.py
```


## Quantum Information exam

For the oral exam of Quantum Information, I presented the experiments made alongside the theory from the papers using [this presentation](presentation/barren-plateaus-presentation.pdf).

## References

<a name="McClean">[1]</a> J.R. McClean, S. Boixo, V.N. Smelyanskiy et al. *Barren plateaus in quantum neural network training landscapes.* (2018)

<a name="Grant">[2]</a> E. Grant, L. Wossnig, M. Ostaszewski and M. Benedetti. *An initialization strategy for addressing barren plateaus in parametrized quantum circuits.* (2019)

<a name="TFQ">[3]</a> TensorFlow Quantum tutorials. *Barren plateaus* https://www.tensorflow.org/quantum/tutorials/barren_plateaus?hl=en

<a name="Cirq">[4]</a> Cirq library. https://quantumai.google/cirq

## Bonus: Cirq examples

The directory [cirq_examples](cirq_examples) contains some examples about using the *Cirq* library. They have been useful to understand how the library works.
