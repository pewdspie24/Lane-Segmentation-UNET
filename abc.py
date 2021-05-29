import numpy as np

def random_vector(m):
    return np.random.randint(11, size = m)

def random_matrix(n):
    return np.random.randint(11, size = (n,n))

print(random_vector(5))
print(random_matrix(5))