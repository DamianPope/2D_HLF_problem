# 2D_HLF_problem
Code that implements the quantum circuit that solves the 2D Hidden Linear Function (2D HLF) problem.
The problem was introduced in the following paper:<br>
  &nbsp; &nbsp; &nbsp; Quantum advantage with shallow circuits<br>
  &nbsp; &nbsp; &nbsp; Sergey Bravyi, David Gosset, Robert Koenig<br>
  &nbsp; &nbsp; &nbsp; https://arxiv.org/abs/1704.00690<br>
  &nbsp; &nbsp; &nbsp; Science Vol. 362, Issue 6412, pp. 308-311 (2018)<br>
It involves finding a simplifying linear function that's "hidden" within a more complicated quadratic form.

The paper proves that the quantum circuit solves the problem more efficiently than the best possible classical circuit. This proves the existence of quantum advantage for this problem.

The code in this repository uses Xanadu's Pennylane QML library: https://pennylane.ai/ and IBM's Qiskit SDK: https://qiskit.org/

After I had written the code, I learnt of closely related code on the same problem for [Qiskit](https://qiskit.org/documentation/stubs/qiskit.circuit.library.HiddenLinearFunction.html?highlight=hiddenlinearfunction#qiskit.circuit.library.HiddenLinearFunction) & [Cirq](https://github.com/fedimser/quant_comp/blob/master/2D%20Hidden%20Linear%20Function.ipynb).



