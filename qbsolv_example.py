from dwave_qbsolv import QBSolv
import neal
import itertools
import random

qubo_size = 500
subqubo_size = 30
Q = {t: random.uniform(-1, 1) for t in itertools.product(range(qubo_size), repeat=2)}
sampler = neal.SimulatedAnnealingSampler()
response = QBSolv().sample_qubo(Q, solver=sampler, solver_limit=subqubo_size)
print("energies=" + str(list(response.data_vectors['energy'])))
