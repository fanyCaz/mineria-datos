import preprocessing as pre
import numpy as np
import pandas as pd

matrix = pd.read_csv('hoax.csv')

matrix = np.array(matrix)
print(matrix)

center = 1.56
pre.calculate_distances(matrix,center,88)
pre.bias(matrix[:,0],center)
pre.bias(matrix[:,1],88)


