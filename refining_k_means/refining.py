import numpy as np
from numpy import ndarray as nd
from main import kmeans, kmeansMod

def refine(initial_start_point, data, k, num_subsamples=2):
  cm = {}
  flatten_data = nd.flatten(data)
  #limit = np.random.randint(1,len(data)/2)
  max_rows = 5
  for i in range(num_subsamples):
    #s_i = np.random.choice(flatten_data, limit)
    rand_start = np.random.randint(1,max_rows)
    s_i = data[rand_start:rand_start+max_rows]
    print(f'subsample {s_i} stat {rand_start}')
    cm_i = kmeansMod(initial_start_point,s_i,k)
    cm = cm + cm_i

  return

