from preprocessing import calculate_distances, normalize, new_centroids
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys

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

def read_data(file_name):
  matrix = []
  debug = True
  try:
    matrix = pd.read_csv(file_name)
    # complete numerical values
    if debug:
      matrix = matrix[['REGION-CENTROID-COL','REGION-CENTROID-ROW','REGION-PIXEL-COUNT']]
    else:
      matrix = matrix.select_dtypes('number')
  except:
    raise FileNotFoundError
  return matrix

matrix = read_data('segmentation_paper.csv')
matrix = np.array(matrix)
#print(matrix)
norm_matrix = normalize(matrix)
centers = np.array([[0,0,0],[1,1,1]])
#centers = np.array([[0,0,0],[1,1,1],[0.5,0.5,0.5]])
#centers = np.array([[0,0,0],[1,1,1],[0,1,1],[1,0,1]])
#centers = np.array([[0,0,0],[1,1,1],[0,1,1],[1,0,0],[0.5,0.5,0.5]])
#centers = np.array([[0,0,0],[1,1,1],[0,1,1],[1,0,1],[0.5,0.5,0.5],[0.2,0.2,0.2]])

kmeans(norm_matrix,centers)
