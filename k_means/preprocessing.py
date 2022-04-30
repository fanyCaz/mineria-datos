import numpy as np
import math

def calculate_distances(matrix, centers) -> list:
  distances = []
  for idx,row in enumerate(matrix):
    temp = []
    for center in centers:
      value = math.sqrt( math.pow(center[0] - row[0],2) + math.pow(center[1] - row[1],2) )
      temp.append(value)
    distances.append( temp )
  distances = np.array(distances)
  return distances

def new_centroids(distances,centers,original,j_original):
  new_centroids_calc = [] 
  sums_centers = np.zeros( len(centers) )
  j_obj = 0
  for idx,row in enumerate(distances):
    temp = []
    min_value = min(row)
    j_obj += min_value
    for jdx,c_temp in enumerate(centers):
      temp.append( 1 if min_value == row[jdx] else 0 )
      sums_centers[jdx] += ( 1 if min_value == row[jdx] else 0 )
    new_centroids_calc.append( temp )
  new_centroids_calc = np.array(new_centroids_calc) 

  if math.isclose(j_original,j_obj,rel_tol=0.00001):
    return centers,j_original, new_centroids_calc

  new_centroids = []
  for idx,center in enumerate(centers):
    centroid = []
    centroid.append( (center[0] + sum(list(map(lambda x,y: x*y,original[:,0],new_centroids_calc[:,idx])))) / (sums_centers[idx] +1)  )
    centroid.append( (center[1] + sum(list(map(lambda x,y: x*y,original[:,1],new_centroids_calc[:,idx])))) / (sums_centers[idx] +1)  )
    new_centroids.append( centroid )
  new_centroids = np.array(new_centroids)
  return new_centroids,j_obj, new_centroids_calc

def slope(column: list,y0: float = 0, y1: float = 1):
  x0 = min(column)
  x1 = max(column)
  m = (y1-y0)/(x1 - x0)
  b = y0-m*x0
  return m,b

def normalize(matrix: list):
  slopes = [ slope(matrix[:,idx]) for idx in range(len(matrix[0])) ]
  for idx,row in enumerate(matrix):
    for jdx,element in enumerate(row): 
      value = slopes[jdx][0]*element + slopes[jdx][1]
      matrix[idx][jdx] = value
  return matrix

