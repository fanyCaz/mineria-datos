from preprocessing import calculate_distances, normalize, new_centroids
from refining import refine, kmeans, kmeansMod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from utils import generador

# una vez obtenido el scalar de distortion entonces ya se puede aplicar kmeans a toda la base de datos, 
# esta parte primero hay que guardar los centros en un txt y de ahi los leo para kmeans general
def read_data(file_name):
  matrix = []
  debug = True
  try:
    matrix = pd.read_csv(file_name)
    if debug:
      matrix = matrix[['REGION-CENTROID-COL','REGION-CENTROID-ROW','REGION-PIXEL-COUNT']]
    else:
      # complete numerical values
      matrix = matrix.select_dtypes('number')
  except:
    raise FileNotFoundError
  return matrix

def input_normalized(question: str, extra_constraint=None):
  centers_iniciales = input(question)
  try:
    if int(centers_iniciales) < 0:
      return input_normalized(question)
    else:
      if extra_constraint:
        if int(centers_iniciales) < extra_constraint[0] or int(centers_iniciales) > 17:
          print(f"Ingresa un número entre {extra_constraint[0]} y {17}")
          return input_normalized(question, extra_constraint)
      return int(centers_iniciales)
  except:
    print('Ingresa un para los centros , porfavor')
    return input_normalized(question)
  
matrix = read_data('segmentation_paper.csv')
matrix = np.array(matrix,dtype = 'float64')
length_df = len(matrix[0])
number_centroids = input_normalized('Ingresa el numero de centros: ',[1,length_df])

norm_matrix = normalize(matrix)
#centers = np.array([[0,0,0],[1,1,1]])
#centers = np.array([[0,0,0],[1,1,1],[0.5,0.5,0.5]])
#centers = np.array([[0,0,0],[1,1,1],[0,1,1],[1,0,1]])
#centers = np.array([[0,0,0],[1,1,1],[0,1,1],[1,0,0],[0.5,0.5,0.5]])

centers = generador(number_centroids, length_df)
k = len(centers)
#kmeans(norm_matrix,centers)
refine(centers,norm_matrix,k)

