import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.tree import export_graphviz
from six import StringIO  
import pydotplus


df = pd.read_csv('registros.csv')

df = df.drop('ID', axis=1)

casado_map = {True: 1, False: 0}

encoder = preprocessing.LabelEncoder()

encoder = encoder.fit(['Primaria', 'Secundaria', 'Preparatoria', 'Superior'])

one_hot = preprocessing.OneHotEncoder(handle_unknown = 'ignore')

df['Casad@'] = df['Casad@'].map(casado_map)

df['Escolaridad'] = encoder.transform(df['Escolaridad'])

enc_df = pd.DataFrame(one_hot.fit_transform(df[['Municipio']]).toarray())

df = df.join(enc_df)

salida_map = {'Aprobado': 1, 'Denegado': 0}

df['Credito'] = df['Credito'].map(salida_map)

X = df.drop('Credito', axis=1)

Y = df['Credito']

clf = DecisionTreeClassifier(criterion="entropy")

clf = clf.fit(X, Y)

feature_cols = ['Edad', 'Casad@', 'Escolaridad', 'Municipio']

export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_png('diabetes.png')


