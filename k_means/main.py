from preprocessing import calculate_distances, normalize, new_centroids
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys

def centroids_belonging(belonging: list):
  identifiers = ['A','B','C','D','E','F','G','H','I','J'] 
  elements = {}
  for idx,element in enumerate(belonging):
    index = np.argmax(element)
    try:
      elements[index].append(identifiers[idx])
    except:
      elements[index] = []
      elements[index].append(identifiers[idx])
  return elements

def kmeans(norm_matrix,centers):
  j_objective = 0
  j_ant = sys.maxsize
  belonging = []
  while True:
    distances = calculate_distances(norm_matrix,centers)
    centers, j_objective, belonging = new_centroids(distances,centers,norm_matrix,j_ant)
    if math.isclose(j_objective,j_ant,rel_tol=0.00001):
      break
    else:
      j_ant = j_objective
  print(f"Para {len(centers)} centros, el mejor objetivo logrado fue : {j_objective}")
  print("Los centros son:")
  print(centers)

  elements = centroids_belonging(belonging)
  print("Elementos de cada centroide")
  print(elements)

matrix = pd.read_csv('hoax.csv')
matrix = np.array(matrix)
norm_matrix = normalize(matrix)
#centers = np.array([[0,0],[1,1]])
#centers = np.array([[0,0],[1,1],[0.5,0.5]])
#centers = np.array([[0,0],[1,1],[0,1],[1,0]])
#centers = np.array([[0,0],[1,1],[0,1],[1,0],[0.5,0.5]])
centers = np.array([[0,0],[1,1],[0,1],[1,0],[0.5,0.5],[0.2,0.2]])

kmeans(norm_matrix,centers)
