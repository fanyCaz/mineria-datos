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

def new_centroids(distances,centers):
  min_indexes_distances = np.argmin(distances,axis=1)
  print( np.amin(distances,axis=1) )
  min_distances = np.amin(distances,axis=1)
  temp = []
  for idx,row in enumerate(distances):
    for c_temp in enumerate(centers):
      value = [ min_distances[idx] == c_temp]
      print( value) 

  sum_indexes = sum(min_indexes_distances)
  return

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

