import numpy as np
import sys

def assign_groups(data, centroids):
  group_vector = []
  for idx, element in enumerate(data):
    min_distance = sys.maxsize
    distances = []
    centroid_id = 0
    for c_id,centroid in enumerate(centroids):
      distance = np.linalg.norm( element - centroid ) 
      if distance < min_distance:
        min_distance = distance
        centroid_id = c_id
    group_vector.append(centroid_id+1)
  return group_vector

def update_centroids(data, group_vector, centroids):
  new_centroids = []
  for c_id, centroid in enumerate(centroids):
    centroid_group = np.zeros(len(data[0]))
    count = 0
    for d_idx,element in enumerate(data):
      if group_vector[d_idx] == (c_id+1):
        centroid_group = centroid_group + element
        count += 1
    group_average = centroid_group/count
    new_centroids.append( group_average )
  return new_centroids 

def calculate_objective(data, group_vector, centroids):
  j_objective = 0
  for d_idx, element in enumerate(data):
    for c_idx, centroid in enumerate(centroids):
      if group_vector[d_idx] == (c_idx+1):
        j_objective += np.linalg.norm( element - centroid )**2
  j_objective = j_objective/len(data)
  return j_objective


def run(data, centroids):
  j_objective_vector = []
  converge = False
  while converge == False:
    group_vector = assign_groups(data, centroids)
    new_centroids = update_centroids(data, group_vector, centroids)
    j_objective = calculate_objective(data, group_vector, new_centroids)
    j_objective_vector.append(j_objective)
    if np.linalg.norm( np.array(new_centroids) - np.array(centroids) ) < 1e-6:
      converge = True
    else:
      centroids = new_centroids
  return new_centroids, group_vector, j_objective_vector

