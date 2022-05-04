import numpy as np
import pandas as pd
from itertools import permutations

data = pd.read_csv("ejemplo.csv")

variables = data.columns.values
rule_combinations = list(permutations(variables,2))
print( rule_combinations)

grouping = {}
for i in variables:
  if  data[i].dtypes == 'int64':
    q1 = data[i].quantile(0.25)
    q2 = data[i].quantile(0.5)
    q3 = data[i].quantile(0.75)
    q4 = data[i].max()
    grouping[i] = {'g_type': 'number', 'elements': [q1,q2,q3,q4] }
  else:
    grouping[i] = {'g_type': 'category', 'elements': list(set(data[i])) }

print(grouping)

print( rule_combinations[0] )

f =  rule_combinations[0][0] 
print(data[f])

# si el grupo es int64, entonces hacer variable bool de es menor a q1? y asi por las 4, y luego si no es entonces variable bool de es igual el valor a el elemento 0?
# después solo contar los que cumplan con ambos, con uno de cada uno y el total de elementos
# pregunta: los cuantiles valen como suficiente para dividir los datos?
"""
for combinacion in perm:
  combinacion =list(combinacion)
  datos_comb=datos[combinacion]
  contador=contador+1
"""

