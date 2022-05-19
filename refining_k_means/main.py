from preprocessing import calculate_distances, normalize, new_centroids
from refining import refine, kmeans, kmeansMod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# una vez obtenido el scalar de distortion entonces ya se puede aplicar kmeans a toda la base de datos, 
# esta parte primero hay que guardar los centros en un txt y de ahi los leo para kmeans general
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
matrix = np.array(matrix,dtype = 'float64')
norm_matrix = normalize(matrix)
#centers = np.array([[0,0,0],[1,1,1]])
#centers = np.array([[0,0,0],[1,1,1],[0.5,0.5,0.5]])
#centers = np.array([[0,0,0],[1,1,1],[0,1,1],[1,0,1]])
#centers = np.array([[0,0,0],[1,1,1],[0,1,1],[1,0,0],[0.5,0.5,0.5]])
centers = np.array([[0,0,0],[1,1,1],[0,1,1],[1,0,1],[0.5,0.5,0.5],[0.2,0.2,0.2]])
k = len(centers)
#kmeans(norm_matrix,centers)
refine(centers,norm_matrix,k)

