from preprocessing import calculate_distances, normalize, new_centroids
from refining import refine, kmeans, kmeansMod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime
from utils import generador, input_normalized, imprimir_matriz, print_norm_matrix

def read_data(file_name, number_columns,type_sep):
  matrix = []
  debug = False
  try:
    matrix = pd.read_csv(file_name, sep=type_sep)
    data_types_count = matrix['variety'].unique().size
    if number_columns == 0:
      # complete numerical values
      #matrix = matrix.select_dtypes('number')
      matrix = matrix.iloc[:,:-1]
    else:
      matrix = matrix.iloc[:,1:number_columns]
      #matrix = matrix[['REGION-CENTROID-COL','REGION-CENTROID-ROW','REGION-PIXEL-COUNT','SHORT-LINE-DENSITY-5','SHORT-LINE-DENSITY-2','VEDGE-MEAN']]
  except:
    raise FileNotFoundError
  return matrix,data_types_count

print("Refinamiento de centros para k-means")
print("Antes de comenzar, responde lo siguiente:")
menu_selection = input_normalized('1. Quiero usar toda la base de datos\n 2. Quiero solo usar algunas columnas\n->',[1,2])
if menu_selection == 1:
  num_columns = 0
elif menu_selection == 2:
  num_columns = input_normalized('Escribe cuántas columnas quieres usar solo se pueden usar del 2 al 7: ',[1,7])

matrix,data_types_count = read_data('seeds_dataset.txt', num_columns,'\t')
matrix = np.array(matrix,dtype = 'float64')
length_df = len(matrix[0])
max_rows_sample = 10

print("Se ha leído el dataset..")
number_centroids = input_normalized(f'Ingresa el numero de centros a usar: Solo se pueden usar de 2 a {data_types_count}: ',[1,data_types_count])

norm_matrix = normalize(matrix)
print_norm_matrix('matriz_normalizada',norm_matrix)
centers = generador(number_centroids, length_df)
k = len(centers)

print("Primero se refinarán los centros")
number_subsamples = 10
refined_centers, _, _ = refine(centers,norm_matrix,k,number_subsamples,max_rows_sample)
imprimir_matriz(f'centros_refinados_{number_centroids}',refined_centers)
time_now = datetime.now().strftime("%d%B%Y_%I_%M")
elements, centers, j_objective, belonging,s_elements = kmeans(norm_matrix,refined_centers)
imprimir_matriz(f'elementos_refinados_{time_now}',s_elements,f'Objetivo logrado: {j_objective}')
print(f'Objetivo logrado con refinamiento: {j_objective}')
elements, centers, j_objective, belonging,s_elements = kmeans(norm_matrix,centers)
print(f'Objetivo logrado sin refinamiento: {j_objective}')
imprimir_matriz(f'elementos_no_refinados_{time_now}',s_elements,f'Objetivo logrado: {j_objective}')

