from preprocessing import calculate_distances, normalize, new_centroids
from refining import refine, kmeans, kmeansMod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from utils import generador, input_normalized, imprimir_matriz


# una vez obtenido el scalar de distortion entonces ya se puede aplicar kmeans a toda la base de datos,
# esta parte primero hay que guardar los centros en un txt y de ahi los leo para kmeans general
def read_data(file_name, n_cols=2):
    matrix = []
    matrix = pd.read_csv(file_name)
    if n_cols == 2:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW']]
    elif n_cols == 3:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT']]
    elif n_cols == 4:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5']]
    elif n_cols == 5:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5',
                         'SHORT-LINE-DENSITY-2']]
    elif n_cols == 6:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5',
                         'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN']]
    elif n_cols == 7:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5',
                         'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN', 'VEDGE-SD']]
    elif n_cols == 8:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5',
                         'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN']]
    elif n_cols == 9:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5',
                         'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD']]
    elif n_cols == 10:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5',
                         'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN']]
    elif n_cols == 11:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5',
                         'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN',
                         'RAWRED-MEAN']]
    elif n_cols == 12:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5',
                         'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN',
                         'RAWRED-MEAN', 'RAWBLUE-MEAN']]
    elif n_cols == 13:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5',
                         'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN',
                         'RAWRED-MEAN', 'RAWBLUE-MEAN', 'RAWGREEN-MEAN']]
    elif n_cols == 14:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5',
                         'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN',
                         'RAWRED-MEAN', 'RAWBLUE-MEAN', 'RAWGREEN-MEAN', 'EXRED-MEAN']]
    elif n_cols == 15:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5',
                         'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN',
                         'RAWRED-MEAN', 'RAWBLUE-MEAN', 'RAWGREEN-MEAN', 'EXRED-MEAN', 'EXBLUE-MEAN']]

    elif n_cols == 16:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5',
                         'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN',
                         'RAWRED-MEAN', 'RAWBLUE-MEAN', 'RAWGREEN-MEAN', 'EXRED-MEAN', 'EXBLUE-MEAN', 'EXGREEN-MEAN']]

    elif n_cols == 17:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5',
                         'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN',
                         'RAWRED-MEAN', 'RAWBLUE-MEAN', 'RAWGREEN-MEAN', 'EXRED-MEAN', 'EXBLUE-MEAN', 'EXGREEN-MEAN',
                         'VALUE-MEAN']]
    elif n_cols == 18:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5',
                         'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN',
                         'RAWRED-MEAN', 'RAWBLUE-MEAN', 'RAWGREEN-MEAN', 'EXRED-MEAN', 'EXBLUE-MEAN', 'EXGREEN-MEAN',
                         'VALUE-MEAN', 'SATURATION-MEAN']]

    elif n_cols == 19:
        matrix = matrix[['REGION-CENTROID-COL', 'REGION-CENTROID-ROW', 'REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5',
                         'SHORT-LINE-DENSITY-2', 'VEDGE-MEAN', 'VEDGE-SD', 'HEDGE-MEAN', 'HEDGE-SD', 'INTENSITY-MEAN',
                         'RAWRED-MEAN', 'RAWBLUE-MEAN', 'RAWGREEN-MEAN', 'EXRED-MEAN', 'EXBLUE-MEAN', 'EXGREEN-MEAN',
                         'VALUE-MEAN', 'SATURATION-MEAN', 'HUE-MEAN']]
    return matrix


column_names = ["id", "num_centroids", "num_variables", "result_pre_refining", "result_post_refining", "distortion_cm",
                "distortion_fm"]
results = pd.DataFrame()

for num_run in range(0, 100):
    for num_centroids in range(2, 11):
        for num_variables in range(2, 20):
            matrix = read_data('segmentation_paper.csv', num_variables)
            matrix = np.array(matrix, dtype='float64')
            length_df = len(matrix[0])
            max_rows_sample = 10
            norm_matrix = normalize(matrix)
            centers = generador(num_centroids, length_df)
            k = len(centers)
            print("---------------------------------------------------------------------------------------------------")
            print(f"Empezando con {num_centroids} y {num_variables}")
            print("Refinamiento de k-means")
            print("Primero se refinarán los centros")
            number_subsamples = 10
            refined_centers = refine(centers, norm_matrix, k, number_subsamples, max_rows_sample)
            # imprimir_matriz('centros_refinados',refined_centers)
            elements, centers, j_objective, belonging = kmeans(norm_matrix, centers)
            print(f'Objetivo logrado sin refinamiento: {j_objective}')
            distortion = distortion()
            print(f"Distorsion sin refinamiento: ")
            elements, centers, j_objective, belonging = kmeans(norm_matrix, refined_centers)
            print(f'Objetivo logrado con refinamiento: {j_objective}')
