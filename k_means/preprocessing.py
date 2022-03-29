import numpy as np
import math

def calculate_distances(matrix, center: float, bias: float) -> list:
  distances = list(map(lambda index: math.sqrt( math.pow(index[0]-center,2) + math.pow(index[1]-bias,2) ) ,matrix))
  return distances

def bias(column: list, center: float) -> list:
  biases = list(map(lambda value: math.pow(value-center,2),column ))
  print(biases)
  return biases
