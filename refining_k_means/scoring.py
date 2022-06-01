import numpy as np
import pandas as pd
import math

def information_gain(matrix, file):
  data = pd.read_csv(file)
  classes = data['LABEL'].unique()
  elements_per_class = list(map(lambda cl: len(data.query(f'LABEL == "{cl}"')), classes))
  number_data = len(data)
  print(f' c - {classes} - {number_data}')
  entropy = 0
  for epc in elements_per_class:
    entropy += (epc/number_data) * math.log(epc/number_data)
  print(f"Entropía de la base de datos : {entropy}")


  return

information_gain([],'segmentation_paper.csv')