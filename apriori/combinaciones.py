import numpy as np
import pandas as pd
from itertools import permutations
from apriori import apriori
from metric_rules import support, confidence, lift

def read_data():
  data = pd.read_csv("registros.csv")
  variables = data.columns.values
  return data, variables[1:]

def create_combinations(data,variables):
  variable_combinations = list(permutations(variables,2))

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

  return variable_combinations,grouping

def generate_rules(data,combinations,grouping):
  n_if = 0
  n_then = 0
  n_both = 0
  total_elements = len(data)
  metrics_a = []
  metrics = {}
  counter = 0
  for idx,variable in enumerate(combinations):
    rule_if = variable[0]
    rule_then = variable[1]
    #print(f'if {rule_if} then {rule_then}')
    min_q = 0
    for kdx,element_if in enumerate(grouping[rule_if]["elements"]):
      #print(f'{rule_if} -> {element_if}')
      if kdx > 0:
        min_q = grouping[rule_if]["elements"][kdx-1]
      if grouping[rule_if]["g_type"] == "number":
        query_if = f'{rule_if} > {min_q} & {rule_if} <= {element_if}'
      else:
        query_if = f'{rule_if} == "{element_if}"'
      if_data = data.query(query_if)
      n_if = len(if_data)
      min_q_j = 0
      for jdx,element_then in enumerate(grouping[rule_then]["elements"]):
        #print(f'{rule_then} -> {element_then}')
        if jdx > 0:
          min_q_j = grouping[rule_then]["elements"][jdx -1]
        if grouping[rule_then]["g_type"] == "number":
          query_then = f'{rule_then} > {min_q_j} & {rule_then} <= {element_then}'
        else:
          query_then = f'{rule_then} == "{element_then}"' 
        then_data = if_data.query(query_then)
        n_then = len(data.query(query_then))
        n_both = len(then_data)
        s = support(total_elements, n_both)
        c = confidence(n_both,n_if)
        l = lift(total_elements,n_if,n_then,n_both)
        metrics_a.append( {'rule': f'{query_if}:{query_then}', 's': s, 'c': c, 'l': l} )
    metrics_a.append(metrics)
  print(metrics_a)
  return metrics_a

def main():
  data, variables = read_data()
  variable_combinations, grouping = create_combinations(data,variables)
  metrics = generate_rules(data,variable_combinations,grouping)
  minsup = 0.15
  apriori(metrics, minsup, data,variables)

if __name__  == '__main__':
  main()

