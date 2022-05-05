import numpy as np
import pandas as pd
from itertools import permutations

data = pd.read_csv("ejemplo.csv")

variables = data.columns.values

variable_combinations = list(permutations(variables,2))
print( variable_combinations)

grouping = {}
for i in variables:
  if  data[i].dtypes == 'int64':
    q1 = data[i].quantile(0.25)
    q2 = data[i].quantile(0.5)
    q3 = data[i].quantile(0.75)
    q4 = data[i].max()
    grouping[i] = {'g_type': 'number', 'elements': [q1,q2,q3,q4] }
  else:
    grouping[i] = {'g_type': 'category', 'elements': data[i].unique() }

rule_if =  variable_combinations[0][0] 
rule_then = variable_combinations[0][1]

for idx,variable in enumerate(variable_combinations):
  rule_if = variable[0]
  rule_then = variable[1]
  print(f'if {rule_if} then {rule_then}')
  print(f'{rule_if} = {grouping[rule_if]["elements"][0]}')

  for i in grouping[rule_if]["elements"]:
    print(i)
    for j in grouping[rule_then]["elements"]:
      print(j)

  if grouping[rule_if]["g_type"] == "number":
    if_data = data.query(f'{rule_if} <= {grouping[rule_if]["elements"][0]}')
    # llamar a función que haga ciclo entre todos los elementos
  else:
    if_data = data.query(f'{rule_if} == "{grouping[rule_if]["elements"][0]}"')
  if grouping[rule_then]["g_type"] == "number":
    then_data = if_data.query(f'{rule_then} <= {grouping[rule_then]["elements"][0]}')
  else:
    then_data = if_data.query(f'{rule_then} == "{grouping[rule_then]["elements"][0]}"')
  print( len(then_data) )


# llamar a función que haga ciclo entre todos los elementos
# después solo contar los que cumplan con ambos, con uno de cada uno y el total de elementos
# pregunta: los cuantiles valen como suficiente para dividir los datos? Si es suficiente para las variables que son como continuas, se dejará así para el sake del programa

