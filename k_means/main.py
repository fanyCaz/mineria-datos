from preprocessing import calculate_distances, bias, slope, adjustment
from kmeans import run 
import numpy as np
import pandas as pd

matrix = pd.read_csv('hoax.csv')

matrix = np.array(matrix)

center = 1.56
calculate_distances(matrix,center,88)
bias(matrix[:,0],center)
bias(matrix[:,1],88)
slope(matrix[:,0])
m,b = slope(matrix[:,1])
adjusted = adjustment(m,b,matrix[:,1])

number_klusters = 3

data = np.concatenate([[0.3*np.random.randn(2) for i in range(200)],[[1,1] + 0.3*np.random.randn(2) for i in range(200)], [[1,-1]+0.3*np.random.randn(2) for i in range(200)]])

centroids = data[:number_klusters]
result = run(data, centroids)

print( result )
