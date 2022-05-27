import numpy as np
np.set_printoptions(threshold=np.inf)

def imprimir_matriz(nombre, matriz):
    with open(f"{nombre}.txt", "w+") as archi:
        archi.writelines(f"{nombre}\n")
        archi.writelines("\n")
        archi.writelines("Matriz:\n")
        archi.writelines(f"{matriz}")

def generador(numero_centros):
  centros=np.random.rand(numero_centros)
  return centros.tolist()


