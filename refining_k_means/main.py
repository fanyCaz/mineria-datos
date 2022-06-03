from preprocessing import calculate_distances, normalize, new_centroids
from refining import refine, kmeans, kmeansMod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from datetime import datetime
from utils import generador, input_normalized, imprimir_matriz, print_norm_matrix
# 'types_count': dataset.columns[2:].size, 'data': dataset
#  'types_count': dataset['variety'].unique().size, 'data': dataset.iloc[:,:-1]
def get_data_specifics(dataset_name):
  datasets = {
    'seeds_dataset': { 'csv': 'dataset/seeds_dataset.txt', 'sep': '\t' },
    'abs_work': { 'csv': 'dataset/abs_work.csv', 'sep': ';' }
  }
  return datasets[dataset_name]

def select_columns(number_columns, matrix, dataset_name):
  datasets = {
    'seeds_dataset': { 'types_count': matrix.columns[:-1].size, 'data': matrix.iloc[:,:-1] },
    'abs_work': { 'types_count': matrix.columns[2:].size, 'data': matrix.iloc[:,2:] }
  }
  data = datasets[dataset_name]['data']
  if number_columns == 0:
    matrix = data
    types_count = datasets[dataset_name]['types_count']
  else:
    matrix = data.iloc[:,:number_columns]
    types_count = matrix.columns.size
  return matrix, types_count

def read_data(file_name):
  matrix = []
  try:
    meta_data = get_data_specifics(file_name)
    matrix = pd.read_csv(meta_data['csv'], sep=meta_data['sep'])
  except:
    raise FileNotFoundError
  return matrix, matrix.columns.size

time_now = datetime.now().strftime("%d%B%Y_%I_%M")
print("Refinamiento de centros para k-means")
dataset_name = 'seeds_dataset'
matrix, length_df = read_data(dataset_name)

print("Se ha leído el dataset..")
print("Antes de comenzar, responde lo siguiente:")
menu_selection = input_normalized('1. Quiero usar toda la base de datos\n2. Quiero solo usar algunas columnas\n->',[1,2])

if menu_selection == 1:
  num_columns = 0
elif menu_selection == 2:
  num_columns = input_normalized(f'Escribe cuántas columnas quieres usar solo se pueden usar del 2 al {length_df}: ',[2,length_df])

matrix, data_types_count = select_columns(num_columns, matrix, dataset_name)

matrix = np.array(matrix,dtype = 'float64')

norm_matrix = normalize(matrix)
max_rows_sample = 10
print_norm_matrix('matriz_normalizada',norm_matrix)
number_centroids = input_normalized(f'Ingresa el numero de centros a usar: Solo se pueden usar de 2 a {data_types_count}: ',[2,data_types_count])
number_coordinates = len(matrix[0])
centers = generador(number_centroids, number_coordinates)
k = len(centers)

print("Primero se refinarán los centros")
number_subsamples = 10
refined_centers, _, _ = refine(centers,norm_matrix,k,number_subsamples,max_rows_sample)

imprimir_matriz(f'centros_refinados_{number_centroids}_{time_now}',refined_centers)
elements, centers, j_objective, belonging,s_elements = kmeans(norm_matrix,refined_centers)
imprimir_matriz(f'elementos_refinados_{time_now}',s_elements,f'Objetivo logrado: {j_objective}')
print(f'Objetivo logrado con refinamiento: {j_objective}')
elements, centers, j_objective, belonging,s_elements = kmeans(norm_matrix,centers)
print(f'Objetivo logrado sin refinamiento: {j_objective}')
imprimir_matriz(f'elementos_no_refinados_{time_now}',s_elements,f'Objetivo logrado: {j_objective}')

