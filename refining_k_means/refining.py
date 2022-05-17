from preprocessing import calculate_distances, normalize, new_centroids
import numpy as np
from numpy import ndarray as nd
import sys
import math

def refine(initial_start_point, data, k, num_subsamples=2):
  cm = {1}
  flatten_data = nd.flatten(data)
  print(f'centros iniciales {initial_start_point}')
  #limit = np.random.randint(1,len(data)/2)
  max_rows = 5
  #for i in range(num_subsamples):
  for i in range(2):
    #s_i = np.random.choice(flatten_data, limit)
    rand_start = np.random.randint(1,max_rows)
    s_i = data[rand_start:rand_start+max_rows]
    print(f'subsample {s_i} stat {rand_start}')
    cm_i = kmeansMod(initial_start_point,s_i,k)
    cm = cm.union(cm_i)

  return

def kmeans(norm_matrix,centers):
  j_objective = 0
  j_ant = sys.maxsize
  belonging = []
  count = 0
  while True:
    count+=1
    distances = calculate_distances(norm_matrix,centers)
    centers, j_objective, belonging = new_centroids(distances,centers,norm_matrix,j_ant)
    if math.isclose(j_objective,j_ant,rel_tol=0.0001):
      break
    else:
      j_ant = j_objective
  print(f"Para {len(centers)} centros, el mejor objetivo logrado fue : {j_objective}; el número de iteraciones fueron: {count}")
  print("Los centros son:")
  print(centers)

  elements = centroids_belonging(belonging)
  print("Elementos de cada centroide")
  print(elements)

def kmeansMod(start_point,sample,k):
  j_objective = 0
  j_ant = sys.maxsize
  belonging = []
  count = 0
  centers = start_point
  while True:
    count+=1
    distances = calculate_distances(sample,centers)
    centers, j_objective, belonging = new_centroids(distances,centers,sample,j_ant)
    if math.isclose(j_objective,j_ant,rel_tol=0.0001):
      print("Pertenencia")
      print(belonging)
      print("Los centros ANTES DE  son:")
      print(centers)
      empty_cluster = any([ sum(belonging[:,i]) == 0 for i in range(len(belonging[0])) ])
      if empty_cluster:
        print(f'distancias \n{distances}')
        # deben ser los maximos
        mini = np.max(distances)
        farthest_distances = np.argmax(distances,axis=1)
        print(f'minims {mini} y {farthest_distances}')
        print(f'del sample {sample[farthest_distances[0]]}')
        for idx, center in enumerate(centers):
          if sum(belonging[:,idx]) == 0:
            centers[idx] = sample[farthest_distances[idx]]
      break
    else:
      j_ant = j_objective
  print(f"Para {len(centers)} centros, el mejor objetivo logrado fue : {j_objective}; el número de iteraciones fueron: {count}")
  print("Los centros son:")
  print(centers)

  elements = centroids_belonging(belonging)
  print("Elementos de cada centroide")
  print(elements)  
  return elements, centers

def centroids_belonging(belonging: list):
  identifiers = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V']
  elements = {}
  print(f'cantidad {len(belonging)}')
  for idx,element in enumerate(belonging):
    centroid_number = np.argmax(element)
    try:
      elements[centroid_number].append(idx)
    except:
      elements[centroid_number] = []
      elements[centroid_number].append(idx)
  return elements
