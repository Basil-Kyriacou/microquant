import torch
from microquant.gradvec import StateVector

def test_initialize():
    phi = StateVector(2)
    condition = torch.all(phi.state == torch.tensor([1,0,0,0]))
    assert condition
test_initialize()

def test_hadamard():
    phi = StateVector(2)
    phi.Hadamard()
    condition = torch.all(torch.isclose(phi.state, torch.ones(2 ** 2, dtype=torch.cfloat) / 2))
    assert condition
test_hadamard()

def test_bell():
    phi = StateVector(2)
    phi.Hadamard(0)
    phi.CNOT([0,1])
    condition1 = torch.all(torch.isclose(phi.state, 
                torch.tensor([1,0,0,1], dtype=torch.cfloat) / torch.sqrt(torch.tensor(2))))
    condition2 = torch.all(torch.isclose(phi.measure(), torch.tensor([1,0,0,1]) /2))
    assert condition1 and condition2
test_bell()

def test_cnot():
    phi = StateVector(2)
    phi.set_state(random=True)
    init_state = phi.state
    phi.CNOT([0,1])
    condition1 = torch.all(phi.state == init_state[:,torch.tensor([0,1,3,2])])
    phi.CNOT([1,0])
    condition2 = torch.all(phi.state == init_state[:,torch.tensor([0,2,3,1])])
    phi.CNOT([0,1])
    condition3 = torch.all(phi.state == init_state[:,torch.tensor([0,2,1,3])])
    assert condition1 and condition2 and condition3
test_cnot()

def test_rotation():
    phi = StateVector(1)
    theta = 0.345
    phi.RZ(theta)
    condition = torch.isclose(phi.state[0,0],torch.exp(-1j * torch.tensor(theta) / 2))
    assert condition
test_rotation()