from preprocessing import calculate_distances, normalize, slope, new_centroids
from kmeans import run 
import numpy as np
import pandas as pd
from refining import refine

matrix = pd.read_csv('hoax.csv')
matrix = np.array(matrix)

norm_matrix = normalize(matrix)

centers = np.array([[0,0],[1,1]])

distances = calculate_distances(norm_matrix,centers)
new_centroids(distances,centers)
#for center in centers:
#  distances.append( calculate_distances(norm_matrix,center) )


#result = calculate_distances(matrix,center,88)
#print(result)
#bias(matrix[:,0],centers[0])
#bias(matrix[:,1],88)

#m,b = slope(matrix[:,0])
#norm_data1 = normalize(matrix[:,0],m,b)
#m,b = slope(matrix[:,1])
#norm_data2 = normalize(matrix[:,1],m,b)

#m = np.array([norm_data1,norm_data2])

#d = calculate_distances(m,centers[0])
#print( d )
#d = calculate_distances(m,centers[1])
#print( d )

#number_klusters = 4

# our take
#data = np.concatenate([[0.3*np.random.randn(2) for i in range(200)],[[1,1] + 0.3*np.random.randn(2) for i in range(200)], [[1,-1]+0.3*np.random.randn(2) for i in range(200)]])

#centroids = data[:number_klusters]
#result = run(data, centroids)

#print( result )

# refinenment

#refine(1,data,number_klusters,2)

