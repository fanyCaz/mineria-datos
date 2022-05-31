from preprocessing import calculate_distances, normalize, new_centroids
from refining import refine, kmeans, kmeansMod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from utils import generador, input_normalized, imprimir_matriz

# una vez obtenido el scalar de distortion entonces ya se puede aplicar kmeans a toda la base de datos, 
# esta parte primero hay que guardar los centros en un txt y de ahi los leo para kmeans general
def read_data(file_name):
  matrix = []
  debug = True
  try:
    matrix = pd.read_csv(file_name)
    if debug:
      matrix = matrix[['REGION-CENTROID-COL','REGION-CENTROID-ROW','REGION-PIXEL-COUNT','SHORT-LINE-DENSITY-5','SHORT-LINE-DENSITY-2','VEDGE-MEAN']]
    else:
      # complete numerical values
      matrix = matrix.select_dtypes('number')
  except:
    raise FileNotFoundError
  return matrix

matrix = read_data('segmentation_paper.csv')
matrix = np.array(matrix,dtype = 'float64')
length_df = len(matrix[0])
max_rows_sample = 10

number_centroids = input_normalized('Ingresa el numero de centros: ',[1,max_rows_sample ])

norm_matrix = normalize(matrix)

centers = generador(number_centroids, length_df)
k = len(centers)

print("Refinamiento de k-means")
print("Primero se refinarán los centros")
number_subsamples = 10
refined_centers = refine(centers,norm_matrix,k,number_subsamples,max_rows_sample)
#imprimir_matriz('centros_refinados',refined_centers)

elements, centers, j_objective, belonging = kmeans(norm_matrix,refined_centers)
print(f'Objetivo logrado con refinamiento: {j_objective}')
elements, centers, j_objective, belonging = kmeans(norm_matrix,centers)
print(f'Objetivo logrado sin refinamiento: {j_objective}')

