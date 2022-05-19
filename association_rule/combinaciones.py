import numpy as np
import pandas as pd
from itertools import permutations

def support(total,n_both):
  return n_both/total

def confidence(n_both,n_if):
  return n_both/n_if

def lift(total,n_if,n_then,n_both):
  return (n_both/total) / ( (n_if/total) * (n_then/total) )

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


n_if = 0
n_then = 0
n_both = 0
total_elements = len(data)
for idx,variable in enumerate(variable_combinations):
  rule_if = variable[0]
  rule_then = variable[1]
  print(f'if {rule_if} then {rule_then}')
  min_q = 0
  metrics = {}
  for idx,element_if in enumerate(grouping[rule_if]["elements"]):
    print(f'{rule_if} -> {element_if}')
    if idx > 0:
      min_q = grouping[rule_if]["elements"][idx-1]
    query = f'{rule_if} > {min_q} & {rule_if} <= {element_if}' if grouping[rule_if]["g_type"] == "number" else f'{rule_if} == "{element_if}"' 
    if_data = data.query(query)
    n_if = len(if_data)
    min_q_j = 0
    for jdx,element_then in enumerate(grouping[rule_then]["elements"]):
      print(f'{rule_then} -> {element_then}')
      if jdx > 0:
        min_q_j = grouping[rule_then]["elements"][jdx -1]
      query = f'{rule_then} > {min_q_j} & {rule_then} <= {element_then}' if grouping[rule_then]["g_type"] == "number" else f'{rule_then} == "{element_then}"' 
      then_data = if_data.query(query)
      n_then = len(data.query(query))
      n_both = len(then_data)
      s = support(total_elements, n_both)
      c = confidence(n_both,n_if)
      l = lift(total_elements,n_if,n_then,n_both)
      metrics[f'{rule_if}-{rule_then}'] = {'s': s, 'c': c, 'l': l}
      print(metrics)

