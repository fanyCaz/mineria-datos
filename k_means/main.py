from preprocessing import calculate_distances, normalize, slope, new_centroids
from kmeans import run 
import numpy as np
import pandas as pd
import math
import sys
from refining import refine

matrix = pd.read_csv('hoax.csv')
matrix = np.array(matrix)

norm_matrix = normalize(matrix)

centers = np.array([[0,0],[1,1]])

j_objective = 0
j_ant = sys.maxsize
while True:
  distances = calculate_distances(norm_matrix,centers)
  centers, j_objective = new_centroids(distances,centers,norm_matrix,j_ant)
  if math.isclose(j_objective,j_ant,rel_tol=0.00001):
    break
  else:
    j_ant = j_objective

print(f"Para {len(centers)} centros, el mejor objetivo logrado fue : {j_objective}")
print("Los centros son:")
print(centers)

centers = np.array([[0,0],[1,1],[0.5,0.5]])
j_objective = 0
j_ant = sys.maxsize
while True:
  distances = calculate_distances(norm_matrix,centers)
  centers, j_objective = new_centroids(distances,centers,norm_matrix,j_ant)
  if True:
    break
  if math.isclose(j_objective,j_ant,rel_tol=0.00001):
    break
  else:
    j_ant = j_objective
print(f"Para {len(centers)} centros, el mejor objetivo logrado fue : {j_objective}")
print("Los centros son:")
print(centers)

