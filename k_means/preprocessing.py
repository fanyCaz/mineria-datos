import numpy as np
import math

def calculate_distances(matrix, center: float, bias: float) -> list:
  distances = list(map(lambda index: math.sqrt( math.pow(index[0]-center,2) + math.pow(index[1]-bias,2) ) ,matrix))
  return distances

def bias(column: list, center: float) -> list:
  biases = list(map(lambda value: math.pow(value-center,2),column ))
  return biases

def slope(column: list,y0: int = 0, y1: int = 1):
  x0 = min(column)
  x1 = max(column)
  m = (y1-y0)/(x1 - x0)
  b = -m*x0
  return m,b

def adjustment(m: int,b: int, originals: list):
  adjusted = list(map(lambda original: m*original+b,originals))
  return adjusted


