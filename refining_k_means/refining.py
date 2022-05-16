import numpy as np
from numpy import ndarray as nd
from kmeans import run

# say J subsamples

def refine(initial_start_point, data, k, num_subsamples=2):
  print(data)
  cm = {}
  flatten_data = nd.flatten(data)
  limit = np.random.randint(1,len(data)/2)
  for i in range(num_subsamples):
    s_i = np.random.choice(flatten_data, limit)
    print(s_i)
    print("result")
    print(s_i[:k])
    result = run(np.array(s_i),s_i[:k])
    print(result)

  return
