from preprocessing import calculate_distances, normalize, new_centroids, calculate_distortion
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
  print("primeros resultados")
  print(cm)
  centers = cm[0]['centers']
  for j in range(num_subsamples):
    elements, centers, j_obj, belonging = kmeans(cm[j]['elements'],centers)
    save = {'elements': s_i, 'centers': centers}
    fms.append(save)
  print("resultados despues")
  print(fms)

  print("Distortion")
  distortion(fms,cm)
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
      elements = centroids_belonging(belonging, distances)
      empty_cluster = len(elements) != len(centers)
      if empty_cluster:
        centers = update_empty_centers(centers,belonging,elements,sample)
      break
    else:
      j_ant = j_objective
  elements = centroids_belonging(belonging, distances)
  return elements, centers, j_objective, belonging

def update_empty_centers(centers, belonging,elements,sample):
  farthest_distance = []
  # find farthest element from non empty clusters
  farthest_element = 0
  max_distance = 0
  for idx,element in enumerate(elements):
    max_distance_idx = np.argmax(list(map(lambda values: values['distance'],elements[element] )))
    current_distance = elements[element][max_distance_idx]['distance']
    if current_distance > max_distance:
      farthest_element = sample[elements[element][max_distance_idx]['idx']]
      max_distance = current_distance

  for idx, center in enumerate(centers):
    if sum(belonging[:,idx]) == 0:
      centers[idx] = farthest_element
      break
  return centers

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
  dis = []
  for jdx,elements in enumerate(fm):
    dis.append( calculate_distances(fm[jdx]['elements'],cm[jdx]['centers']) )
  calculate_distortion(fm[0]['elements'],fm[0]['centers'])
  calculate_distortion(cm[0]['elements'],cm[0]['centers'])

