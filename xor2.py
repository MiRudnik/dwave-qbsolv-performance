import itertools

import dimod
import dwavebinarycsp
from dwave.system import EmbeddingComposite
from dwave.system.samplers import DWaveSampler
from dwavebinarycsp.factories.constraint import gates


def construct_xor_gates_problem(num_inputs):
    if num_inputs < 2:
        num_inputs = 2
    csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)
    csp.add_constraint(gates.xor_gate(['q0', 'q1', 'out0'], name='XOR0'))
    for i in range(2, num_inputs):
        csp.add_constraint(gates.xor_gate(['q{}'.format(i), 'out{}'.format(i - 2), 'out{}'.format(i - 1)],
                                          name='XOR{}'.format(i - 1)))
    csp.add_constraint(lambda x: x, ['out{}'.format(num_inputs - 2)])   # last output must be True
    return csp


def construct_xor_problem(size):
    if size < 2:
        size = 2
    xor_constraints = [p for p in list(itertools.product([0, 1], repeat=size)) if (p.count(1) % 2 == 1)]
    csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)
    csp.add_constraint(dwavebinarycsp.Constraint.
                       from_configurations(xor_constraints, ['q{}'.format(i) for i in range(size)],
                                           dwavebinarycsp.BINARY, name='XOR'))
    return csp


def get_score(resp):
    hits = 0
    for sample, _, num_occurrences, _ in resp.data():
        correct = False
        for k, v in sample.items():
            if k.startswith('q') and v == 1:
                correct = not correct
        if correct:
            hits += num_occurrences
    return hits/reads


reads = 5000
csp = construct_xor_gates_problem(5)
bcm = dwavebinarycsp.stitch(csp, max_graph_size=100)

for c in csp.constraints:
    print(c)    # should be only one for construct_xor_problem

print(bcm)  # prints coefficients
sampler = EmbeddingComposite(DWaveSampler())    # automatic minor embedding
response = sampler.sample(bcm, num_reads=reads)
print(response)
print("Score: " + str(get_score(response)))
