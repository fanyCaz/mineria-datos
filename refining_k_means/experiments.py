import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import hamming

from preprocessing import normalize
from refining import refine, kmeans
from utils import generador


def idx_matrix(matrix):
    # return 1-D array of matrix values in (row_idx, col_idx, val) format
    return [(r, c, val) for r, row in enumerate(matrix)
            for c, val in enumerate(row)]


def find_minima(indexed_vals, limit=0):
    # return array of indexed matrix values whose row and col indexes are unique
    minima = []
    rows = set()
    cols = set()
    for row, col, val in indexed_vals:
        if row not in rows and col not in cols:
            minima.append((row, col, val))
            if limit and len(minima) == limit:
                # optional optimization if you want to break off early
                # after you've found a value for every row
                break
            rows.add(row)
            cols.add(col)
    return minima


def sort_by_val(indexed_vals):
    # return indexed_vals sorted by original matrix value
    return sorted(indexed_vals, key=lambda x: x[2], reverse=True)


def sort_by_col(indexed_vals):
    # return indexed_vals sorted by row index
    return sorted(indexed_vals, key=lambda x: x[1])


def strip_indices(indexed_vals):
    # return a 1-D array with row and col index removed
    return [v[2] for v in indexed_vals]


# https://stackoverflow.com/questions/68719305/find-minimum-for-every-row-in-array-with-unique-columns
def find_maxima_by_col(matrix):
    # put it all together
    indexed = idx_matrix(matrix)
    indexed = sort_by_val(indexed)
    minima = find_minima(indexed)
    minima = sort_by_col(minima)
    return minima


# simple function for calculating how similar are two vectors
def distance(A, B):
    B_unique_sorted, B_idx = np.unique(B, return_index=True)
    B_in_A_bool = np.in1d(B_unique_sorted, A, assume_unique=True)
    return sum(B_in_A_bool)


# una vez obtenido el scalar de distortion entonces ya se puede aplicar kmeans a toda la base de datos,
# esta parte primero hay que guardar los centros en un txt y de ahi los leo para kmeans general
def read_data(file_name):
    matrix = []
    matrix = pd.read_csv(file_name)
    matrix = matrix[['compactness', 'circularity', 'distance_circularity', 'radius_ratio', 'pr_axis_aspect_ratio',
                     'max_length_aspect_ratio', 'scatter_ratio', 'elongatedness', 'pr_axis_rectangularity',
                     'max_length_rectangularity', 'scaled_variance_major', 'scaled_variance_minor',
                     'acaled_radius_gyration', 'skewness_about_major', 'skewness_about_minor', 'kurtosis_about_minor',
                     'kurtosis_about_major', 'hollows_ratio']]
    return matrix


column_names = ["id", "num_centroids", "num_variables", "result_pre_refining", "result_post_refining", "distortion_cm",
                "distortion_fm"]
lines = []

resultados_error_van_pre = []
resultados_acc_van_pre = []
resultados_error_saab_pre = []
resultados_acc_saab_pre = []
resultados_error_bus_pre = []
resultados_acc_bus_pre = []
resultados_error_opel_pre = []
resultados_acc_opel_pre = []


resultados_error_van_post = []
resultados_acc_van_post = []
resultados_error_saab_post = []
resultados_acc_saab_post = []
resultados_error_bus_post = []
resultados_acc_bus_post = []
resultados_error_opel_post = []
resultados_acc_opel_post = []
results = pd.DataFrame()
index = 1
for num_run in range(0, 5):
    for num_centroids in range(2, 6):
        num_variables = 18
        matrix = read_data('dataset\\vehicle_s.csv')
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
        elements, centers, objective_pre, belonging, s_elements = kmeans(norm_matrix, centers,
                                                                         list(range(0, len(matrix))))
        if num_centroids == 4:
            ceros1 = np.zeros(846).astype(int)
            if len(s_elements.keys()) == 4:
                # van = 0
                # saab = 1
                # bus = 2
                # opel = 3

                df = pd.read_csv('dataset\\vehicle_s.csv')
                salida = df['class'].to_numpy().astype(int)
                salida_0 = np.where(df['class'] == 0)
                salida_1 = np.where(df['class'] == 1)
                salida_2 = np.where(df['class'] == 2)
                salida_3 = np.where(df['class'] == 3)

                cluster_0 = s_elements[0]
                cluster_1 = s_elements[1]
                cluster_2 = s_elements[2]
                cluster_3 = s_elements[3]

                # calcular distancia del cluster 0

                dist_cluster_0_class_0 = distance(cluster_0, salida_0)
                dist_cluster_0_class_1 = distance(cluster_0, salida_1)
                dist_cluster_0_class_2 = distance(cluster_0, salida_2)
                dist_cluster_0_class_3 = distance(cluster_0, salida_3)

                dists_cluster_0 = [dist_cluster_0_class_0, dist_cluster_0_class_1, dist_cluster_0_class_2,
                                   dist_cluster_0_class_3]

                # calcular distancia del cluster 1

                dist_cluster_1_class_0 = distance(cluster_1, salida_0)
                dist_cluster_1_class_1 = distance(cluster_1, salida_1)
                dist_cluster_1_class_2 = distance(cluster_1, salida_2)
                dist_cluster_1_class_3 = distance(cluster_1, salida_3)

                dists_cluster_1 = [dist_cluster_1_class_0, dist_cluster_1_class_1, dist_cluster_1_class_2,
                                   dist_cluster_1_class_3]

                # calcular distancia del cluster 2

                dist_cluster_2_class_0 = distance(cluster_2, salida_0)
                dist_cluster_2_class_1 = distance(cluster_2, salida_1)
                dist_cluster_2_class_2 = distance(cluster_2, salida_2)
                dist_cluster_2_class_3 = distance(cluster_2, salida_3)

                dists_cluster_2 = [dist_cluster_2_class_0, dist_cluster_2_class_1, dist_cluster_2_class_2,
                                   dist_cluster_2_class_3]

                # calcular distancia del cluster 3

                dist_cluster_3_class_0 = distance(cluster_3, salida_0)
                dist_cluster_3_class_1 = distance(cluster_3, salida_1)
                dist_cluster_3_class_2 = distance(cluster_3, salida_2)
                dist_cluster_3_class_3 = distance(cluster_3, salida_3)

                dists_cluster_3 = [dist_cluster_3_class_0, dist_cluster_3_class_1, dist_cluster_3_class_2,
                                   dist_cluster_3_class_3]

                dist_matrix = [dists_cluster_0, dists_cluster_1, dists_cluster_2, dists_cluster_3]

                indices = find_maxima_by_col(dist_matrix)
                real_clusters = {}
                for indice in indices:
                    if indice[0] == 0:
                        real_clusters[0] = indice[1]
                    elif indice[0] == 1:
                        real_clusters[1] = indice[1]
                    elif indice[0] == 2:
                        real_clusters[2] = indice[1]
                    elif indice[0] == 3:
                        real_clusters[3] = indice[1]

                real_cluster_0_idx = real_clusters[0]
                real_cluster_1_idx = real_clusters[1]
                real_cluster_2_idx = real_clusters[2]
                real_cluster_3_idx = real_clusters[3]

                idxs = [real_cluster_0_idx, real_cluster_1_idx, real_cluster_2_idx, real_cluster_3_idx]

                for key in sorted(s_elements.keys()):
                    for elem in s_elements[key]:
                        ceros1[elem] = idxs[key]

                cm = confusion_matrix(salida, ceros1)
                print("Matriz de confusion sin refinar: ")
                print(cm)
                tp_van_pre = cm[0][0]
                fn_van_pre = cm[0][1] + cm[0][2] + cm[0][3]  # same row
                fp_van_pre = cm[1][0] + cm[2][0] + cm[3][0]  # same col
                tn_van_pre = cm[1][1] + cm[1][2] + cm[1][3] + cm[2][1] + cm[2][2] + cm[2][3] + cm[3][1] + cm[3][2] + \
                             cm[3][3]
                N = tp_van_pre + fn_van_pre + fp_van_pre + tn_van_pre
                error_van_pre = (fp_van_pre + fn_van_pre) / N
                acc_van_pre = (tp_van_pre + tn_van_pre) / N

                resultados_error_van_pre.append(error_van_pre)
                resultados_acc_van_pre.append(acc_van_pre)

                tp_saab_pre = cm[1][1]
                fn_saab_pre = cm[1][0] + cm[1][2] + cm[1][3]  # same row
                fp_saab_pre = cm[0][1] + cm[2][1] + cm[3][1]  # same col
                tn_saab_pre = cm[0][0] + cm[0][2] + cm[0][3] + cm[2][0] + cm[2][2] + cm[2][3] + cm[3][0] + cm[3][1] + \
                              cm[3][2]
                N = tp_saab_pre + fn_saab_pre + fp_saab_pre + tn_saab_pre
                error_saab_pre = (fp_saab_pre + fn_saab_pre) / N
                acc_saab_pre = (tp_saab_pre + tn_saab_pre) / N

                resultados_error_saab_pre.append(error_saab_pre)
                resultados_acc_saab_pre.append(acc_saab_pre)

                tp_bus_pre = cm[2][2]
                fn_bus_pre = cm[2][0] + cm[2][1] + cm[2][3]  # same row
                fp_bus_pre = cm[0][2] + cm[1][2] + cm[3][2]  # same col
                tn_bus_pre = cm[0][0] + cm[0][1] + cm[0][3] + cm[1][0] + cm[1][1] + cm[1][3] + cm[3][0] + cm[3][1] + \
                             cm[3][3]
                N = tp_bus_pre + fn_bus_pre + fp_bus_pre + tn_bus_pre
                error_bus_pre = (fp_bus_pre + fn_bus_pre) / N
                acc_bus_pre = (tp_bus_pre + tn_bus_pre) / N

                resultados_error_bus_pre.append(error_bus_pre)
                resultados_acc_bus_pre.append(acc_bus_pre)

                tp_bus_pre = cm[2][2]
                fn_bus_pre = cm[2][0] + cm[2][1] + cm[2][3]  # same row
                fp_bus_pre = cm[0][2] + cm[1][2] + cm[3][2]  # same col
                tn_bus_pre = cm[0][0] + cm[0][1] + cm[0][3] + cm[1][0] + cm[1][1] + cm[1][3] + cm[3][0] + cm[3][1] + \
                             cm[3][3]
                N = tp_bus_pre + fn_bus_pre + fp_bus_pre + tn_bus_pre
                error_bus_pre = (fp_bus_pre + fn_bus_pre) / N
                acc_bus_pre = (tp_bus_pre + tn_bus_pre) / N

                resultados_error_bus_pre.append(error_bus_pre)
                resultados_acc_bus_pre.append(acc_bus_pre)

                tp_opel_pre = cm[3][3]
                fn_opel_pre = cm[3][0] + cm[3][1] + cm[3][2]  # same row
                fp_opel_pre = cm[0][3] + cm[1][3] + cm[2][3]  # same col
                tn_opel_pre = cm[0][0] + cm[0][1] + cm[0][2] + cm[1][0] + cm[1][1] + cm[1][2] + cm[2][0] + cm[2][1] + \
                              cm[2][3]
                N = tp_opel_pre + fn_opel_pre + fp_opel_pre + tn_opel_pre
                error_opel_pre = (fp_opel_pre + fn_opel_pre) / N
                acc_opel_pre = (tp_opel_pre + tn_opel_pre) / N

                resultados_error_opel_pre.append(error_opel_pre)
                resultados_acc_opel_pre.append(acc_opel_pre)

        # print(f'Objetivo logrado sin refinamiento: {objective_pre}')
        # print(f"Distorsion sin refinamiento: {dist1}")
        elements, centers, j_objective, belonging, s_elements2 = kmeans(norm_matrix, refined_centers,
                                                                        list(range(0, len(matrix))))

        if num_centroids == 4:
            ceros2 = np.zeros(846).astype(int)
            if len(s_elements2.keys()) == 4:
                # van = 0
                # saab = 1
                # bus = 2
                # opel = 3

                df = pd.read_csv('dataset\\vehicle_s.csv')
                salida = df['class'].to_numpy().astype(int)
                salida_0 = np.where(df['class'] == 0)
                salida_1 = np.where(df['class'] == 1)
                salida_2 = np.where(df['class'] == 2)
                salida_3 = np.where(df['class'] == 3)

                cluster_0_ref = s_elements2[0]
                cluster_1_ref = s_elements2[1]
                cluster_2_ref = s_elements2[2]
                cluster_3_ref = s_elements2[3]

                # calcular distancia del cluster 0

                dist_cluster_0_class_0_ref = distance(cluster_0_ref, salida_0)
                dist_cluster_0_class_1_ref = distance(cluster_0_ref, salida_1)
                dist_cluster_0_class_2_ref = distance(cluster_0_ref, salida_2)
                dist_cluster_0_class_3_ref = distance(cluster_0_ref, salida_3)

                dists_cluster_0_ref = [dist_cluster_0_class_0_ref, dist_cluster_0_class_1_ref,
                                       dist_cluster_0_class_2_ref, dist_cluster_0_class_3_ref]

                # calcular distancia del cluster 1

                dist_cluster_1_class_0_ref = distance(cluster_1_ref, salida_0)
                dist_cluster_1_class_1_ref = distance(cluster_1_ref, salida_1)
                dist_cluster_1_class_2_ref = distance(cluster_1_ref, salida_2)
                dist_cluster_1_class_3_ref = distance(cluster_1_ref, salida_3)

                dists_cluster_1_ref = [dist_cluster_1_class_0_ref, dist_cluster_1_class_1_ref,
                                       dist_cluster_1_class_2_ref, dist_cluster_1_class_3_ref]

                # calcular distancia del cluster 2

                dist_cluster_2_class_0_ref = distance(cluster_2_ref, salida_0)
                dist_cluster_2_class_1_ref = distance(cluster_2_ref, salida_1)
                dist_cluster_2_class_2_ref = distance(cluster_2_ref, salida_2)
                dist_cluster_2_class_3_ref = distance(cluster_2_ref, salida_3)

                dists_cluster_2_ref = [dist_cluster_2_class_0_ref, dist_cluster_2_class_1_ref,
                                       dist_cluster_2_class_2_ref,dist_cluster_2_class_3_ref]

                # calcular distancia del cluster 3

                dist_cluster_3_class_0_ref = distance(cluster_3_ref, salida_0)
                dist_cluster_3_class_1_ref = distance(cluster_3_ref, salida_1)
                dist_cluster_3_class_2_ref = distance(cluster_3_ref, salida_2)
                dist_cluster_3_class_3_ref = distance(cluster_3_ref, salida_3)

                dists_cluster_3_ref = [dist_cluster_3_class_0_ref, dist_cluster_3_class_1_ref,
                                       dist_cluster_3_class_2_ref, dist_cluster_3_class_3_ref]

                dist_matrix_ref = [dists_cluster_0_ref, dists_cluster_1_ref,
                                   dists_cluster_2_ref, dists_cluster_3_ref]

                indices_ref = find_maxima_by_col(dist_matrix_ref)
                real_clusters_ref = {}
                for indice in indices_ref:
                    if indice[0] == 0:
                        real_clusters_ref[0] = indice[1]
                    elif indice[0] == 1:
                        real_clusters_ref[1] = indice[1]
                    elif indice[0] == 2:
                        real_clusters_ref[2] = indice[1]
                    elif indice[0] == 3:
                        real_clusters_ref[3] = indice[1]

                real_cluster_0_idx_ref = real_clusters_ref[0]
                real_cluster_1_idx_ref = real_clusters_ref[1]
                real_cluster_2_idx_ref = real_clusters_ref[2]
                real_cluster_3_idx_ref = real_clusters_ref[3]

                idxs_ref = [real_cluster_0_idx_ref, real_cluster_1_idx_ref, real_cluster_2_idx_ref,
                            real_cluster_3_idx_ref]

                for key in sorted(s_elements2.keys()):
                    for elem in s_elements2[key]:
                        ceros2[elem] = idxs_ref[key]
                cm2 = confusion_matrix(salida, ceros2)
                print("Matriz de confuson refinada: ")
                print(cm2)

                tp_van_post = cm2[0][0]
                fn_van_post = cm2[0][1] + cm2[0][2] + cm2[0][3]  # same row
                fp_van_post = cm2[1][0] + cm2[2][0] + cm2[3][0]  # same col
                tn_van_post = cm2[1][1] + cm2[1][2] + cm2[1][3] + cm2[2][1] + cm2[2][2] + cm2[2][3] + cm2[3][1] + cm2[3][2] + \
                             cm2[3][3]
                N = tp_van_post + fn_van_post + fp_van_post + tn_van_post
                error_van_post = (fp_van_post + fn_van_post) / N
                acc_van_post = (tp_van_post + tn_van_post) / N

                resultados_error_van_post.append(error_van_post)
                resultados_acc_van_post.append(acc_van_post)
                tp_saab_post = cm2[1][1]
                fn_saab_post = cm2[1][0] + cm2[1][2] + cm2[1][3]  # same row
                fp_saab_post = cm2[0][1] + cm2[2][1] + cm2[3][1]  # same col
                tn_saab_post = cm2[0][0] + cm2[0][2] + cm2[0][3] + cm2[2][0] + cm2[2][2] + cm2[2][3] + cm2[3][0] + cm2[3][1] + \
                              cm2[3][2]
                N = tp_saab_post + fn_saab_post + fp_saab_post + tn_saab_post
                error_saab_post = (fp_saab_post + fn_saab_post) / N
                acc_saab_post = (tp_saab_post + tn_saab_post) / N

                resultados_error_saab_post.append(error_saab_post)
                resultados_acc_saab_post.append(acc_saab_post)

                tp_bus_post = cm2[2][2]
                fn_bus_post = cm2[2][0] + cm2[2][1] + cm2[2][3]  # same row
                fp_bus_post = cm2[0][2] + cm2[1][2] + cm2[3][2]  # same col
                tn_bus_post = cm2[0][0] + cm2[0][1] + cm2[0][3] + cm2[1][0] + cm2[1][1] + cm2[1][3] + cm2[3][0] + cm2[3][1] + \
                             cm2[3][3]
                N = tp_bus_post + fn_bus_post + fp_bus_post + tn_bus_post
                error_bus_post = (fp_bus_post + fn_bus_post) / N
                acc_bus_post = (tp_bus_post + tn_bus_post) / N

                resultados_error_bus_post.append(error_bus_post)
                resultados_acc_bus_post.append(acc_bus_post)

                tp_bus_post = cm2[2][2]
                fn_bus_post = cm2[2][0] + cm2[2][1] + cm2[2][3]  # same row
                fp_bus_post = cm2[0][2] + cm2[1][2] + cm2[3][2]  # same col
                tn_bus_post = cm2[0][0] + cm2[0][1] + cm2[0][3] + cm2[1][0] + cm2[1][1] + cm2[1][3] + cm2[3][0] + cm2[3][1] + \
                             cm2[3][3]
                N = tp_bus_post + fn_bus_post + fp_bus_post + tn_bus_post
                error_bus_post = (fp_bus_post + fn_bus_post) / N
                acc_bus_post = (tp_bus_post + tn_bus_post) / N

                resultados_error_bus_post.append(error_bus_post)
                resultados_acc_bus_post.append(acc_bus_post)

                tp_opel_post = cm2[3][3]
                fn_opel_post = cm2[3][0] + cm2[3][1] + cm2[3][2]  # same row
                fp_opel_post = cm2[0][3] + cm2[1][3] + cm2[2][3]  # same col
                tn_opel_post = cm2[0][0] + cm2[0][1] + cm2[0][2] + cm2[1][0] + cm2[1][1] + cm2[1][2] + cm2[2][0] + cm2[2][1] + \
                              cm2[2][3]
                N = tp_opel_post + fn_opel_post + fp_opel_post + tn_opel_post
                error_opel_post = (fp_opel_post + fn_opel_post) / N
                acc_opel_post = (tp_opel_post + tn_opel_post) / N

                resultados_error_opel_post.append(error_opel_post)
                resultados_acc_opel_post.append(acc_opel_post)

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

mean_improv_obj_all = df_all.rel_improv_obj.mean()
mean_improv_dist_all = df_all.rel_improv_dist.mean()
mean_err_van_pre = np.mean(resultados_error_van_pre)
mean_acc_van_pre = np.mean(resultados_acc_van_pre)
mean_err_saab_pre = np.mean(resultados_error_saab_pre)
mean_acc_saab_pre = np.mean(resultados_acc_saab_pre)
mean_err_bus_pre = np.mean(resultados_error_bus_pre)
mean_acc_bus_pre = np.mean(resultados_acc_bus_pre)
mean_err_opel_pre = np.mean(resultados_error_opel_pre)
mean_acc_opel_pre = np.mean(resultados_acc_opel_pre)

mean_err_van_post = np.mean(resultados_error_van_post)
mean_acc_van_post = np.mean(resultados_acc_van_post)
mean_err_saab_post = np.mean(resultados_error_saab_post)
mean_acc_saab_post = np.mean(resultados_acc_saab_post)
mean_err_bus_post = np.mean(resultados_error_bus_post)
mean_acc_bus_post = np.mean(resultados_acc_bus_post)
mean_err_opel_post = np.mean(resultados_error_opel_post)
mean_acc_opel_post = np.mean(resultados_acc_opel_post)

line_van_pre = f"Para la clase Van, antes de refinar centros hay un error promedio de {mean_err_van_pre:.3f} y" \
               f" una accuracy promedio de {mean_acc_van_pre:.3f}"
line_van_post = f"Para la clase Van, despues de refinar centros hay un error promedio de {mean_err_van_post:.3f} y" \
                f" una accuracy promedio de {mean_acc_van_post:.3f}"
line_saab_pre = f"Para la clase Saab, antes de refinar centros hay un error promedio de {mean_err_saab_pre:.3f} y" \
               f" una accuracy promedio de {mean_acc_saab_pre:.3f}"
line_saab_post = f"Para la clase Saab, despues de refinar centros hay un error promedio de {mean_err_saab_post:.3f} y" \
                f" una accuracy promedio de {mean_acc_saab_post:.3f}"
line_bus_pre = f"Para la clase Bus, antes de refinar centros hay un error promedio de {mean_err_bus_pre:.3f} y" \
               f" una accuracy promedio de {mean_acc_bus_pre:.3f}"
line_bus_post = f"Para la clase Bus, despues de refinar centros hay un error promedio de {mean_err_bus_post:.3f} y" \
                f" una accuracy promedio de {mean_acc_bus_post:.3f}"
line_opel_pre = f"Para la clase Opel, antes de refinar centros hay un error promedio de {mean_err_opel_pre:.3f} y" \
               f" una accuracy promedio de {mean_acc_opel_pre:.3f}"
line_opel_post = f"Para la clase Opel, despues de refinar centros hay un error promedio de {mean_err_opel_post:.3f} y" \
                f" una accuracy promedio de {mean_acc_opel_post:.3f}"

line = f"Existe una mejora relativa general del {mean_improv_obj_all:.3f} % para el objetivo y " \
       f"{mean_improv_dist_all:.3f} % para la distorsion"

lines.append(line_van_pre)
lines.append(line_van_post)
lines.append(line_saab_pre)
lines.append(line_saab_post)
lines.append(line_bus_pre)
lines.append(line_bus_post)
lines.append(line_opel_pre)
lines.append(line_opel_post)
lines.append(line)

for num_centroids in range(2, 5):
    df_centroids = df_all.loc[df_all.num_centroids == num_centroids]
    mean_improv_obj = df_centroids.rel_improv_obj.mean()
    mean_improv_dist = df_centroids.rel_improv_dist.mean()
    line = f"Existe una mejora relativa del {mean_improv_obj:.3f} % para el objetivo y " \
           f"{mean_improv_dist:.3f} % para la distorsion para {num_centroids} centroides"
    lines.append(line)

num_variables = 18
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
