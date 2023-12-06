import numpy as np
from microquant.statevec import StateVector

def test_initialize():
    phi = StateVector(2)
    condition = np.all(phi.state == np.array([1,0,0,0]))
    assert condition
test_initialize()

def test_hadamard():
    phi = StateVector(2)
    phi.Hadamard()
    condition = np.all(np.isclose(phi.state, np.ones(2 ** 2) / 2))
    assert condition
test_hadamard()

def test_bell():
    phi = StateVector(2)
    phi.Hadamard(0)
    phi.CNOT([0,1])
    condition1 = np.all(np.isclose(phi.state, np.array([1,0,0,1]) / np.sqrt(2)))
    condition2 = np.all(np.isclose(phi.measure(), np.array([1,0,0,1]) /2))
    assert condition1 and condition2
test_bell()

def test_cnot():
    phi = StateVector(2)
    phi.set_state(random=True)
    init_state = phi.state
    phi.CNOT([0,1])
    condition1 = np.all(phi.state == init_state[np.array([0,1,3,2])])
    phi.CNOT([1,0])
    condition2 = np.all(phi.state == init_state[np.array([0,2,3,1])])
    phi.CNOT([0,1])
    condition3 = np.all(phi.state == init_state[np.array([0,2,1,3])])
    assert condition1 and condition2 and condition3
test_cnot()

def test_rotation():
    phi = StateVector(1)
    theta = 0.345
    phi.RZ(theta)
    condition = np.isclose(phi.state[0],np.exp(-1j * theta / 2))
    assert condition
test_rotation()