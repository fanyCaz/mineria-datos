import pandas as pd

from itertools import permutations

path="/MINERIA_DATOS/LABORATORIO/Practica5/"
datos=pd.read_csv(path+"Data_6columns.csv")

datos=datos[["Income","Age","House_Ownership", "Profession"]]

perma=list(datos.columns)

perm = permutations(perma)

perm= list(perm)

contador=0
for combinacion in perm:
  combinacion =list(combinacion)
  datos_comb=datos[combinacion]
  datos_comb.to_csv(f"/content/drive/MyDrive/MINERIA_DATOS/LABORATORIO/Practica5/combinaciones/combinacion_{contador}.csv", index=False)
  contador=contador+1