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
    matrix = pd.read_csv(file_name, sep='\t')
    if n_cols == 2:
        matrix = matrix[['area', 'perimeter']]
    elif n_cols == 3:
        matrix = matrix[['area', 'perimeter', 'compactness']]
    elif n_cols == 4:
        matrix = matrix[['area', 'perimeter', 'compactness', 'length_kernel']]
    elif n_cols == 5:
        matrix = matrix[['area', 'perimeter', 'compactness', 'length_kernel', 'width_kernel']]
    elif n_cols == 6:
        matrix = matrix[['area', 'perimeter', 'compactness', 'length_kernel', 'assimetry']]
    elif n_cols == 7:
        matrix = matrix[['area', 'perimeter', 'compactness', 'length_kernel', 'assimetry', 'length_kernel_groove']]
    return matrix


column_names = ["id", "num_centroids", "num_variables", "result_pre_refining", "result_post_refining", "distortion_cm",
                "distortion_fm"]
results = pd.DataFrame()
index = 1
for num_run in range(0, 2):
    for num_centroids in range(2, 11):
        for num_variables in range(2, 8):
            matrix = read_data('seeds_dataset.txt', num_variables)
            matrix = np.array(matrix, dtype='float64')
            length_df = len(matrix[0])
            max_rows_sample = 10
            norm_matrix = normalize(matrix)
            centers = generador(num_centroids, length_df)
            k = len(centers)
            # print("---------------------------------------------------------------------------------------------------")
            # print(f"Empezando con {num_centroids} centroides y {num_variables} variables")
            # print("Refinamiento de k-means")
            # print("Primero se refinarán los centros")
            number_subsamples = 10
            refined_centers, dist1, dist2 = refine(centers, norm_matrix, k, number_subsamples, max_rows_sample)
            # imprimir_matriz('centros_refinados',refined_centers)
            elements, centers, objective_pre, belonging = kmeans(norm_matrix, centers)
            # print(f'Objetivo logrado sin refinamiento: {objective_pre}')
            # print(f"Distorsion sin refinamiento: {dist1}")
            elements, centers, j_objective, belonging = kmeans(norm_matrix, refined_centers)
            # print(f'Objetivo logrado con refinamiento: {j_objective}')
            # print(f"Distorsion sin refinamiento: {dist2}")
            row = {"id": index, "num_centroids": num_centroids, "num_variables": num_variables, "result_pre_refining":
                objective_pre, "result_post_refining": j_objective, "distortion_cm": dist1, "distortion_fm": dist2}
            # print(row)
            results = results.append(row, ignore_index=True)
            index = index + 1

relative_improvement_obj = (results['result_pre_refining'] - results['result_post_refining'])
rel_obj = (relative_improvement_obj / results['result_pre_refining']) * 100
relative_improvement_obj = rel_obj
relative_improvement_dist = (results['distortion_cm'] - results['distortion_fm'])
rel_dist = (relative_improvement_dist / results['distortion_cm']) * 100
relative_improvement_dist = rel_dist
rels_obj = pd.DataFrame(relative_improvement_obj)
rels_obj = rels_obj.rename(columns={0: 'rel_improv_obj'})
rels_dist = pd.DataFrame(relative_improvement_dist)
rels_dist = rels_dist.rename(columns={0: 'rel_improv_dist'})

df_all = pd.concat([results, rels_obj, rels_dist], axis=1)

lines = []

mean_improv_obj_all = df_all.rel_improv_obj.mean()
mean_improv_dist_all = df_all.rel_improv_dist.mean()

first_line = f"Existe una mejora relativa general del {mean_improv_obj_all:.3f} % para el objetivo y " \
             f"{mean_improv_dist_all:.3f} % para la distorsion"

lines.append(first_line)

for num_centroids in range(2, 11):
    df_centroids = df_all.loc[df_all.num_centroids == num_centroids]
    mean_improv_obj = df_centroids.rel_improv_obj.mean()
    mean_improv_dist = df_centroids.rel_improv_dist.mean()
    line = f"Existe una mejora relativa del {mean_improv_obj:.3f} % para el objetivo y " \
           f"{mean_improv_dist:.3f} % para la distorsion para {num_centroids} centroides"
    lines.append(line)

for num_variables in range(2, 20):
    df_vars = df_all.loc[df_all.num_variables == num_variables]
    mean_improv_obj = df_vars.rel_improv_obj.mean()
    mean_improv_dist = df_vars.rel_improv_dist.mean()
    line = f"Existe una mejora relativa del {mean_improv_obj:.3f} % para el objetivo y " \
           f"{mean_improv_dist:.3f} % para la distorsion para {num_variables} variables"
    lines.append(line)

with open("resultados.txt", "w") as f:
    for line in lines:
        print(line)
        f.write(line)
        f.write("\n")

figure, axis = plt.subplots(2, 2)
axis[0, 0].plot(results['id'], results['result_pre_refining'])
axis[0, 0].plot(results['id'], results['result_post_refining'], color='red')
axis[0, 0].set_title("Resultados del objetivo")
axis[0, 0].set_xlabel("Iteración")
axis[0, 0].set_ylabel("Valor de objetivo")

axis[0, 1].plot(results['id'], results['distortion_cm'])
axis[0, 1].plot(results['id'], results['distortion_fm'], color='red')
axis[0, 1].set_title("Resultados de la distorsion")
axis[0, 1].set_xlabel("Iteración")
axis[0, 1].set_ylabel("Valor de distorsión")

axis[1, 0].plot(results['id'], df_all['rel_improv_obj'])
axis[1, 0].axhline(y=0.0, color='r', linestyle='-')
axis[1, 0].set_title("% de mejora relativa del objetivo")
axis[1, 0].set_xlabel("Iteración")
axis[1, 0].set_ylabel("% de mejora relativa")

axis[1, 1].plot(results['id'], df_all['rel_improv_dist'])
axis[1, 1].axhline(y=0.0, color='r', linestyle='-')
axis[1, 1].set_title("% de mejora relativa de distorsión")
axis[1, 1].set_xlabel("Iteración")
axis[1, 1].set_ylabel("% de mejora relativa")

plt.show()
