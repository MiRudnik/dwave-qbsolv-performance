import itertools

from dwave.embedding import embed_qubo
from dwave.system.samplers import DWaveSampler
from dwave_networkx import chimera_graph


def create_minor_embedded_xor_qubo(size):
    if not 65 > size > 0 or size % 4 != 0:
        return None
    grid_size = int(size / 4)
    phisical_qubits = grid_size + 1
    bias = (-1 / 3 + grid_size) / phisical_qubits
    # biases
    q = {(8 * 16 * row + 8 * unit + x, 8 * 16 * row + 8 * unit + x): bias
         for x in range(8) for row in range(grid_size) for unit in range(row + 1)}
    # coupling
    for unit in range(grid_size):
        # self connect
        for col in range(unit + 1):
            for x in range(4):
                for y in range(4):
                    q[(8 * 16 * unit + 8 * col + x,
                       8 * 16 * unit + 8 * col + 4 + y)] = -1 if x == y and col == unit else 2 / 3
        # vertical connect
        for row in range(unit + 1, grid_size):
            for x in range(4):
                q[(8 * 16 * row + 8 * unit + x, 8 * 16 * (row + 1) + 8 * unit + x)] = -1
        # horizontal connect
        for col in range(0, unit):
            for x in range(4):
                q[(8 * 16 * unit + 8 * col + x + 4, 8 * 16 * unit + 8 * (col + 1) + x + 4)] = -1
    return q


def create_xor_qubo(size):
    if size < 2:
        size = 2
    q = {(i, i): -1 for i in range(size)}
    for i in range(size):
        for j in range(i + 1, size):
            q[(i, j)] = 2
    return q


def create_embedding(size):
    embedding = {i: set() for i in range(size)}
    grid_size = int(size/4) + 1
    for i in range(size):
        row = int(i / 4)
        embedding[i].add(128 * row + 8 * row + (i % 4))
        embedding[i].add(128 * row + 8 * row + (i % 4) + 4)
        for r in range(row-1, -1, -1):
            embedding[i].add(128 * row + 8 * r + (i % 4) + 4)
        for c in range(row, grid_size-1):
            embedding[i].add(128 * c + 8 * row + (i % 4))
    return embedding


def get_score(resp, size, shots):
    hits = 0
    for sample, _, num_occurrences in resp.data():
        unembedded = {'q{}'.format(i): 0 for i in range(size)}
        correct = False
        # print(str(sample))
        for key, value in sample.items():
            if value == 1:
                x = int(key)
                row = int(x / 128)
                x = x % 128
                col = int(x / 8)
                x = x % 8
                side = int(x / 4)
                x = x % 4
                var = x + 4 * row if side else x + 4 * col
                unembedded['q{}'.format(var)] = 1
        if list(unembedded.values()).count(1) == 1:
            hits += num_occurrences
            correct = True
        # print(str(unembedded) + " " + str(num_occurrences) + " " + str(correct))
    return hits / shots


problem_size = 8
shots = 1000

Q = create_xor_qubo(problem_size)
print("QUBO:")
print(Q)
embedding = create_embedding(problem_size)
print("Embedding:")
print(embedding)

tQ = embed_qubo(Q, embedding, chimera_graph(16),  chain_strength=0.9)
print("Embedded QUBO:")
print(tQ)

# if tQ:
#     response = DWaveSampler().sample_qubo(tQ, num_reads=shots)
#     print("Response:")
#     print(get_score(response, problem_size, shots))
