from preprocessing import calculate_distances, normalize, new_centroids
import numpy as np
from numpy import ndarray as nd
import sys
import math

def refine(initial_start_point, data, k, num_subsamples=1):
  cm = []
  flatten_data = nd.flatten(data)
  print(f'centros iniciales \n{initial_start_point}')
  max_rows = 10
  centers = initial_start_point
  for i in range(num_subsamples):
    rand_start = np.random.randint(1,max_rows)
    s_i = data[rand_start:rand_start+max_rows]
    elements, centers, j_obj, belonging = kmeansMod(centers,s_i,k)
    save = {'elements': s_i, 'centers': centers}
    cm.append(save)
  fms = []
  #print("primeros resultados")
  #print(cm)
  """
  centers = cm[0]['centers']
  for j in range(num_subsamples):
    elements, centers, j_obj, belonging = kmeans(cm[j]['elements'],centers)
    save = {'elements': s_i, 'centers': centers}
    fms.append(save)
  print("resultados despues")
  print(fms)
  """
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

  elements = centroids_belonging(belonging, distances)
  return elements, centers, j_objective, belonging

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
      # creo que verificar empty_cluster no es necesario
      empty_cluster = any([ sum(belonging[:,i]) == 0 for i in range(len(belonging[0])) ])
      if empty_cluster:
        elements = centroids_belonging(belonging, distances)
        farthest_distances = np.argmax(distances,axis=1)
        farthest_distance = []
        for w in elements:
          # de todos estos elementos, obtener el de mayor distancia a su cluster, y obtener la coordenada de ese elemento
          wat =  [ (r['idx'],r['distance']) for r in elements[w] ]
          max_distance = np.argmax([ r['distance'] for r in elements[w] ])
          print(f'{max_distance} y {np.max([r["distance"] for r in elements[w]] )} ')
          print(f'donde anda {w} --{wat} ')
          farthest_distance.append(max_distance)
        for idx, center in enumerate(centers):
          if sum(belonging[:,idx]) == 0:
            centers[idx] = sample[farthest_distances[idx]]
      break
    else:
      j_ant = j_objective

  elements = centroids_belonging(belonging, distances)
  return elements, centers, j_objective, belonging

def centroids_belonging(belonging: list, distances: list):
  identifiers = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V']
  elements = {}
  for idx,element in enumerate(belonging):
    centroid_number = np.argmax(element)
    try:
      elements[centroid_number].append({'idx':idx, 'distance': np.min(distances[idx]) })
    except:
      elements[centroid_number] = []
      elements[centroid_number].append({'idx':idx, 'distance': np.min(distances[idx]) })
  return elements

def distortion(fm,cm):
  for jdx,elements in enumerate(fm):
    calculate_distances(fm[jdx]['elements'],cm[jdx]['centers'])
  calculate_distances(matrix,centers)

