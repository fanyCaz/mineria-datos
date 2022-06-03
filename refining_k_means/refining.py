from preprocessing import calculate_distances, normalize, new_centroids, calculate_distortion
from utils import imprimir_matriz
import numpy as np
from numpy import ndarray as nd
import sys
import math

def refine(initial_start_point, data, k, num_subsamples=4, max_rows=10):
  cm = []
  #print(f'centros iniciales \n{initial_start_point}')
  centers = initial_start_point
  for i in range(num_subsamples):
    s_i = get_subsample(data,max_rows)
    elements, centers, j_obj, belonging, s_elements = kmeansMod(centers,s_i,k)
    save = {'elements': s_i, 'centers': centers}
    cm.append(save)
  #idx_min_distortion,min_distortion = distortion(cm)
  #print(f"minima fm {min_distortion}")
  fms = []
  #print("primeros resultados")
  #print( elements )
  #print(cm)
  for j in range(num_subsamples):
    centers = cm[j]['centers']
    elements, centers, j_obj, belonging, s_elements = kmeans(cm[j]['elements'],centers)
    save = {'elements': cm[j]['elements'], 'centers': centers}
    fms.append(save)
  #print("resultados despues")
  #print(fms)
  #print(f'centros finales \n{centers}')
  #print( elements )
  best_centers, dist1, dist2 = distortion(fms,cm)
  #print(f"minima fm {min_distortion}")
  #best_centers = fms[idx_min_distortion]['centers']
  return best_centers, dist1, dist2

def get_subsample(data,num_elements):
  rand_elements = np.random.choice(data.shape[0], num_elements,replace=False)
  sample = data[rand_elements]
  return sample

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
  elements, s_elements = centroids_belonging(belonging, distances)
  return elements, centers, j_objective, belonging, s_elements

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
      elements, s_elements = centroids_belonging(belonging, distances)
      empty_cluster = len(elements) != len(centers) # asking if not all clusters has elements
      if empty_cluster:
        centers = update_empty_centers(centers,belonging,elements,sample)
      break
    else:
      j_ant = j_objective
  elements, s_elements = centroids_belonging(belonging, distances)
  return elements, centers, j_objective, belonging, s_elements

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
  simplified_elements = {}
  for idx,element in enumerate(belonging):
    centroid_number = np.argmax(element)
    try:
      elements[centroid_number].append({'idx':idx, 'distance': np.min(distances[idx]) })
      simplified_elements[centroid_number].append(idx)
    except:
      elements[centroid_number] = []
      elements[centroid_number].append({'idx':idx, 'distance': np.min(distances[idx]) })
      simplified_elements[centroid_number] = []
      simplified_elements[centroid_number].append(idx)
  return elements, simplified_elements

def distortion(k_solution, smooth_solution):
  distortions_fm = []
  for jdx,elements in enumerate(k_solution):
    distortion = calculate_distortion(k_solution[jdx]['elements'],k_solution[jdx]['centers'])
    distortions_fm.append(distortion)
  min_distortion_fm_idx = np.argmin( distortions_fm )

  distortions_cm = []
  for jdx,elements in enumerate(smooth_solution):
    distortion = calculate_distortion(smooth_solution[jdx]['elements'],smooth_solution[jdx]['centers'])
    distortions_cm.append(distortion)
  min_distortion_cm_idx = np.argmin( distortions_cm )

  min_distortion = 0
  best_centers = []
  if(distortions_cm[min_distortion_cm_idx] < distortions_fm[min_distortion_fm_idx]):
    best_centers = smooth_solution[min_distortion_cm_idx]['centers']
  else:
    best_centers = smooth_solution[min_distortion_fm_idx]['centers']
  imprimir_matriz('distorsion_fm', distortions_fm)
  imprimir_matriz('distorsion_cm', distortions_cm)

  return best_centers, distortions_cm[min_distortion_cm_idx], distortions_fm[min_distortion_fm_idx]
