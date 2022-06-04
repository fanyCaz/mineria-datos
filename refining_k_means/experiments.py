from preprocessing import calculate_distances, normalize, new_centroids
from refining import refine, kmeans, kmeansMod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from utils import generador, input_normalized, imprimir_matriz
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
lines = []

resultados_error_karma_pre = []
resultados_acc_karma_pre = []
resultados_error_rosa_pre = []
resultados_acc_rosa_pre = []
resultados_error_can_pre = []
resultados_acc_can_pre = []

resultados_error_karma_post = []
resultados_acc_karma_post = []
resultados_error_rosa_post = []
resultados_acc_rosa_post = []
resultados_error_can_post = []
resultados_acc_can_post = []
results = pd.DataFrame()
index = 1
for num_run in range(0, 5):
    for num_variables in range(2, 8):
        num_centroids = 3
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
        elements, centers, objective_pre, belonging, s_elements = kmeans(norm_matrix, centers)
        if num_centroids == 3:
            ceros1 = np.zeros(210).astype(int)
            try:
                if len(s_elements.keys()) == 3:
                    for key in sorted(s_elements.keys()):
                        print(key)
                        for elem in s_elements[key]:
                            ceros1[elem] = key + 1
                    df = pd.read_csv('seeds_dataset.txt', sep='\t')
                    salida = df['variety'].to_numpy().astype(int)
                    cm = confusion_matrix(salida, ceros1)
                    print(cm)
                    tp_karma_pre = cm[0][0]
                    fn_karma_pre = cm[0][1] + cm[0][2]  # same row
                    fp_karma_pre = cm[1][0] + cm[2][0]  # same col
                    tn_karma_pre = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]
                    N = tp_karma_pre + fn_karma_pre + fp_karma_pre + tn_karma_pre
                    error_karma_pre = (fp_karma_pre + fn_karma_pre) / N
                    acc_karma_pre = (tp_karma_pre + tn_karma_pre) / N

                    resultados_error_karma_pre.append(error_karma_pre)
                    resultados_acc_karma_pre.append(acc_karma_pre)

                    tp_rosa_pre = cm[1][1]
                    fn_rosa_pre = cm[1][0] + cm[1][2]  # same row
                    fp_rosa_pre = cm[0][1] + cm[0][1]  # same col
                    tn_rosa_pre = cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2]
                    N = tp_rosa_pre + fn_rosa_pre + fp_rosa_pre + tn_rosa_pre
                    error_rosa_pre = (fp_rosa_pre + fn_rosa_pre) / N
                    acc_rosa_pre = (tp_rosa_pre + tn_rosa_pre) / N

                    resultados_error_rosa_pre.append(error_rosa_pre)
                    resultados_acc_rosa_pre.append(acc_rosa_pre)

                    tp_can_pre = cm[2][2]
                    fn_can_pre = cm[2][0] + cm[2][1]  # same row
                    fp_can_pre = cm[0][2] + cm[1][2]  # same col
                    tn_can_pre = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1]
                    N = tp_can_pre + fn_can_pre + fp_can_pre + tn_can_pre
                    error_can_pre = (fp_can_pre + fn_can_pre) / N
                    acc_can_pre = (tp_can_pre + tn_can_pre) / N

                    resultados_error_can_pre.append(error_can_pre)
                    resultados_acc_can_pre.append(acc_can_pre)
            except:
                print("")



        # print(f'Objetivo logrado sin refinamiento: {objective_pre}')
        # print(f"Distorsion sin refinamiento: {dist1}")
        elements, centers, j_objective, belonging, s_elements2 = kmeans(norm_matrix, refined_centers)


        if num_centroids == 3:
            ceros2 = np.zeros(210).astype(int)
            try:
                if len(s_elements2.keys()) == 3:
                    for key in sorted(s_elements2.keys()):
                        for elem in s_elements2[key]:
                            ceros2[elem] = key + 1

                    df = pd.read_csv('seeds_dataset.txt', sep='\t')
                    salida = df['variety'].to_numpy()
                    #print(ceros2)
                    #print(salida)
                    cm2 = confusion_matrix(salida, ceros2)
                    print(cm2)
                    tp_karma_post = cm2[0][0]
                    fn_karma_post = cm2[0][1] + cm2[0][2]  # same row
                    fp_karma_post = cm2[1][0] + cm2[2][0]  # same col
                    tn_karma_post = cm2[1][1] + cm2[1][2] + cm2[2][1] + cm2[2][2]
                    N = tp_karma_post + fn_karma_post + fp_karma_post + tn_karma_post
                    error_karma_post = (fp_karma_post + fn_karma_post) / N
                    acc_karma_post = (tp_karma_post + tn_karma_post) / N

                    resultados_error_karma_post.append(error_karma_post)
                    resultados_acc_karma_post.append(acc_karma_post)

                    tp_rosa_post = cm2[1][1]
                    fn_rosa_post = cm2[1][0] + cm2[1][2]  # same row
                    fp_rosa_post = cm2[0][1] + cm2[0][1]  # same col
                    tn_rosa_post = cm2[0][0] + cm2[0][2] + cm2[2][0] + cm2[2][2]
                    N = tp_rosa_post + fn_rosa_post + fp_rosa_post + tn_rosa_post
                    error_rosa_post = (fp_rosa_post + fn_rosa_post) / N
                    acc_rosa_post = (tp_rosa_post + tn_rosa_post) / N

                    resultados_error_rosa_post.append(error_rosa_post)
                    resultados_acc_rosa_post.append(acc_rosa_post)

                    tp_can_post = cm2[2][2]
                    fn_can_post = cm2[2][0] + cm2[2][1]  # same row
                    fp_can_post = cm2[0][2] + cm2[1][2]  # same col
                    tn_can_post = cm2[0][0] + cm2[0][1] + cm2[1][0] + cm2[1][1]
                    N = tp_can_post + fn_can_post + fp_can_post + tn_can_post
                    error_can_post = (fp_can_post + fn_can_post) / N
                    acc_can_post = (tp_can_post + tn_can_post) / N

                    resultados_error_can_post.append(error_can_post)
                    resultados_acc_can_post.append(acc_can_post)
            except:
                print("")


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

mean_err_kar_pre = np.mean(resultados_error_karma_pre)
mean_acc_kar_pre = np.mean(resultados_acc_karma_pre)
mean_err_ros_pre = np.mean(resultados_error_rosa_pre)
mean_acc_ros_pre = np.mean(resultados_acc_rosa_pre)
mean_err_can_pre = np.mean(resultados_error_can_pre)
mean_acc_can_pre = np.mean(resultados_acc_can_pre)

mean_err_kar_post = np.mean(resultados_error_karma_post)
mean_acc_kar_post = np.mean(resultados_acc_karma_post)
mean_err_ros_post = np.mean(resultados_error_rosa_post)
mean_acc_ros_post = np.mean(resultados_acc_rosa_post)
mean_err_can_post = np.mean(resultados_error_can_post)
mean_acc_can_post = np.mean(resultados_acc_can_post)

line_kar_pre = f"Para la clase Karma, antes de refinar centros hay un error promedio de {mean_err_kar_pre:.3f} y" \
               f" una accuracy promedio de {mean_acc_kar_pre:.3f}"
line_kar_post = f"Para la clase Karma, despues de refinar centros hay un error promedio de {mean_err_kar_post:.3f} y" \
                f" una accuracy promedio de {mean_acc_kar_post:.3f}"
line_ros_pre = f"Para la clase Rosa, antes de refinar centros hay un error promedio de {mean_err_ros_pre:.3f} y" \
               f" una accuracy promedio de {mean_acc_ros_pre:.3f}"
line_ros_post = f"Para la clase Rosa, despues de refinar centros hay un error promedio de {mean_err_ros_post:.3f} y" \
                f" una accuracy promedio de {mean_acc_ros_post:.3f}"
line_can_pre = f"Para la clase Canada, antes de refinar centros hay un error promedio de {mean_err_can_pre:.3f} y" \
               f" una accuracy promedio de {mean_acc_can_pre:.3f}"
line_can_post = f"Para la clase Canada, despues de refinar centros hay un error promedio de {mean_err_can_post:.3f} y" \
                f" una accuracy promedio de {mean_acc_can_post:.3f}"

line = f"Existe una mejora relativa general del {mean_improv_obj_all:.3f} % para el objetivo y " \
       f"{mean_improv_dist_all:.3f} % para la distorsion"

lines.append(line_kar_pre)
lines.append(line_kar_post)
lines.append(line_ros_pre)
lines.append(line_ros_post)
lines.append(line_can_pre)
lines.append(line_can_post)
lines.append(line)

for num_centroids in range(3,4):
    df_centroids = df_all.loc[df_all.num_centroids == num_centroids]
    mean_improv_obj = df_centroids.rel_improv_obj.mean()
    mean_improv_dist = df_centroids.rel_improv_dist.mean()
    line = f"Existe una mejora relativa del {mean_improv_obj:.3f} % para el objetivo y " \
           f"{mean_improv_dist:.3f} % para la distorsion para {num_centroids} centroides"
    lines.append(line)

for num_variables in range(2, 8):
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
