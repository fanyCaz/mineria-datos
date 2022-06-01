from preprocessing import calculate_distances, normalize, new_centroids
from refining import refine, kmeans, kmeansMod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from utils import generador, input_normalized, imprimir_matriz

# una vez obtenido el scalar de distortion entonces ya se puede aplicar kmeans a toda la base de datos, 
# esta parte primero hay que guardar los centros en un txt y de ahi los leo para kmeans general
def read_data(file_name, number_columns):
  matrix = []
  debug = False
  try:
    matrix = pd.read_csv(file_name)
    if number_columns == 0:
      # complete numerical values
      matrix = matrix.select_dtypes('number')
    else:
      matrix = matrix.iloc[:,1:number_columns]
      #matrix = matrix[['REGION-CENTROID-COL','REGION-CENTROID-ROW','REGION-PIXEL-COUNT','SHORT-LINE-DENSITY-5','SHORT-LINE-DENSITY-2','VEDGE-MEAN']]
  except:
    raise FileNotFoundError
  return matrix

print("Refinamiento de centros para k-means")
print("Antes de comenzar, responde lo siguiente:")
menu_selection = input_normalized('1. Quiero usar toda la base de datos\n 2. Quiero solo usar algunas columnas\n->',[1,2])
if menu_selection == 1:
  num_columns = 0
elif menu_selection == 2:
  num_columns = input_normalized('Escribe cuántas columnas quieres usar: ',[1,19])

matrix = read_data('segmentation_paper.csv', num_columns)
matrix = np.array(matrix,dtype = 'float64')
length_df = len(matrix[0])
max_rows_sample = 10

print("Se ha leído el dataset..")
number_centroids = input_normalized('Ingresa el numero de centros a usar: ',[1,max_rows_sample ])

norm_matrix = normalize(matrix)

centers = generador(number_centroids, length_df)
k = len(centers)

print("Primero se refinarán los centros")
number_subsamples = 10
refined_centers, _, _ = refine(centers,norm_matrix,k,number_subsamples,max_rows_sample)
#imprimir_matriz('centros_refinados',refined_centers)

elements, centers, j_objective, belonging = kmeans(norm_matrix,refined_centers)
print(f'Objetivo logrado con refinamiento: {j_objective}')
elements, centers, j_objective, belonging = kmeans(norm_matrix,centers)
print(f'Objetivo logrado sin refinamiento: {j_objective}')

