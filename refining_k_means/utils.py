import numpy as np
np.set_printoptions(threshold=np.inf)

def imprimir_matriz(nombre, matriz, extra_str=None):
  with open(f"{nombre}.txt", "w+") as archi:
    archi.writelines(f"{nombre}\n")
    archi.writelines("\n")
    archi.writelines("Matriz:\n")
    archi.writelines(f"{matriz}")
    if extra_str:
      archi.writelines("\n")
      archi.writelines(extra_str)

def print_norm_matrix(nombre,matrix):
  np.savetxt("matriz_normalizada.csv", matrix, delimiter=",")

def generador(numero_centros, numero_coordenadas):
  centros=np.random.rand(numero_centros, numero_coordenadas)
  return centros

"""
Optional: extra_constraint can receive an array of two elements,
the minimum value and the maximum value like: [0,2]
"""
def input_normalized(question: str, extra_constraint=None):
  centers_iniciales = input(question)
  try:
    if int(centers_iniciales) < 0:
      print('Ingresa un número positivo')
      return input_normalized(question,extra_constraint)
    else:
      if extra_constraint:
        if int(centers_iniciales) < extra_constraint[0] or int(centers_iniciales) > extra_constraint[1]:
          print(f"Ingresa un número entre {extra_constraint[0]} y {extra_constraint[1]}")
          return input_normalized(question, extra_constraint)
      return int(centers_iniciales)
  except:
    print('Ingresa un para los centros , porfavor')
    return input_normalized(question,extra_constraint)
