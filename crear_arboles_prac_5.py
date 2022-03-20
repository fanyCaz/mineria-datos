import pandas as pd
import numpy as np

import math

o_df = pd.read_csv(f'/MINERIA_DATOS/LABORATORIO/Practica5/combinaciones/combinacion_23.csv')
total_instances = len(o_df)

# Ranges for numerical values
# Quantiles to get ranges
i_q1= o_df['Income'].quantile(0.25)
i_q2 = o_df['Income'].quantile(0.5)
i_q3 = o_df['Income'].quantile(0.75)
i_q4 = o_df['Income'].max()
i_quantiles = [i_q1,i_q2,i_q3,i_q4]
a_q1 = o_df['Age'].quantile(0.25)
a_q2 = o_df['Age'].quantile(0.5)
a_q3 = o_df['Age'].quantile(0.75)
a_q4 = o_df['Age'].max()
a_quantiles = [a_q1,a_q2,a_q3,a_q4]
house_own_types = o_df['House_Ownership'].unique()
profession_types = o_df['Profession'].unique()

entropies_incomes = []
entropies_age = []
entropies_house = []
entropies_profession = []

branch_incomes = []
branch_ages = []
branch_hown = []
branch_prof = []

def clean_entropies():
  entropies_incomes = []
  entropies_age = []
  entropies_house = []
  entropies_profession = []

def clean_branches():
  branch_incomes = []
  branch_ages = []
  branch_hown = []
  branch_prof = []

#FUENTE -> INCOME
min = 0
for idx,i_q in enumerate(i_quantiles):
  if idx > 0:
    min = i_quantiles[idx-1]
  branch = o_df.query(f'Income > {min} & Income <= {i_q}')
  branch_incomes.append(branch)
  probability = len(branch) / total_instances
  entropies_incomes.append( -probability*math.log10(probability) )

#INCOME -> AGE
for branch in branch_incomes:
  min = 0
  for idx,a_q in enumerate(a_quantiles):
    if idx > 0:
      min = a_quantiles[idx - 1]
    a_branch = branch.query(f'Age > {min} & Age <= {a_q}')
    branch_ages.append(a_branch)
    if len(branch) > 0:
      probability = len(a_branch) / len(branch)
      if probability > 0:
        entropies_age.append( -probability*math.log10(probability) )
len(entropies_age)

#AGE -> HOUSE
for branch in branch_ages:
  for home in house_own_types:
    h_branch = branch.query(f'House_Ownership == "{home}"')
    branch_hown.append(h_branch)
    if len(branch) > 0:
      probability = len(h_branch) / len(branch)
      if probability > 0:
        entropies_house.append( -probability*math.log10(probability) )
len(entropies_house)

#HOUSE -> PROFESSION
for branch in branch_hown:
  for prof in profession_types:
    p_branch = branch.query(f'Profession == "{prof}"')
    branch_prof.append(p_branch)
    probability = len(p_branch) / len(branch)
    if probability > 0:
      entropies_profession.append( -probability*math.log10(probability) )
len(entropies_profession)

#AGE -> PROFESSION
for branch in branch_ages:
  for prof in profession_types:
    p_branch = branch.query(f'Profession == "{prof}"')
    branch_prof.append(p_branch)
    if len(branch) > 0:
      probability = len(p_branch) / len(branch)
      if probability > 0:
        entropies_profession.append( -probability*math.log10(probability) )
len(entropies_profession)

#PROFESSION -> HOUSE
for branch in branch_prof:
  for home in house_own_types:
    h_branch = branch.query(f'House_Ownership == "{home}"')
    branch_hown.append(h_branch)
    if len(branch) > 0 :
      probability = len(h_branch) / len(branch)
      if probability > 0:
        entropies_house.append( -probability*math.log10(probability) )
len(entropies_house)

#INCOME -> HOUSE
for branch in branch_incomes:
  for home in house_own_types:
    h_branch = branch.query(f'House_Ownership == "{home}"')
    branch_hown.append(h_branch)
    if len(branch) > 0:
      probability = len(h_branch) / len(branch)
      if probability > 0:
        entropies_house.append( -probability*math.log10(probability) )
len(entropies_house)

#HOUSE->AGE
for branch in branch_hown:
  min = 0
  for idx,a_q in enumerate(a_quantiles):
    if idx > 0:
      min = a_quantiles[idx - 1]
    a_branch = branch.query(f'Age > {min} & Age <= {a_q}')
    branch_ages.append(a_branch)
    if len(branch) > 0 :
      probability = len(a_branch) / len(branch)
      if probability > 0:
        entropies_age.append( -probability*math.log10(probability) )
len( entropies_age )

#PROFESSION -> AGE
for branch in branch_prof:
  min = 0
  for idx,a_q in enumerate(a_quantiles):
    if idx > 0:
      min = a_quantiles[idx - 1]
    a_branch = branch.query(f'Age > {min} & Age <= {a_q}')
    branch_ages.append(a_branch)
    if len(branch) > 0 :
      probability = len(a_branch) / len(branch)
      if probability > 0:
        entropies_age.append( -probability*math.log10(probability) )
len( entropies_age )

#INCOME -> PROFESSION
for branch in branch_incomes:
  for prof in profession_types:
    p_branch = branch.query(f'Profession == "{prof}"')
    branch_prof.append(p_branch)
    if len(branch) > 0:
      probability = len(p_branch) / len(branch)
      if probability > 0:
        entropies_profession.append( -probability*math.log10(probability) )
len(entropies_profession)

#FUENTE -> AGE
min = 0
for idx,a_q in enumerate(a_quantiles):
  if idx > 0:
    min = a_quantiles[idx-1]
  a_branch = o_df.query(f'Age > {min} & Age <= {a_q}')
  branch_ages.append(a_branch)
  probability = len(a_branch) / total_instances
  if probability > 0:
    entropies_age.append( -probability*math.log10(probability) )
len(entropies_age)

#AGE -> INCOME
for branch in branch_ages:
  min = 0
  for idx,i_q in enumerate(i_quantiles):
    if idx > 0:
      min = i_quantiles[idx-1]
    i_branch = branch.query(f'Income > {min} & Income <= {i_q}')
    branch_incomes.append(i_branch)
    if len(branch) > 0:
      probability = len(i_branch) / len(branch)
      if probability > 0:
        entropies_incomes.append( -probability*math.log10(probability) )
len(entropies_incomes)

#HOUSE -> INCOME
for branch in branch_hown:
  min = 0
  for idx,i_q in enumerate(i_quantiles):
    if idx > 0:
      min = i_quantiles[idx-1]
    i_branch = branch.query(f'Income > {min} & Income <= {i_q}')
    branch_incomes.append(i_branch)
    if len(branch) > 0:
      probability = len(i_branch) / len(branch)
      if probability > 0:
        entropies_incomes.append( -probability*math.log10(probability) )
len(entropies_incomes)

#PROFESSION -> INCOME
for branch in branch_prof:
  min = 0
  for idx,i_q in enumerate(i_quantiles):
    if idx > 0:
      min = i_quantiles[idx-1]
    i_branch = branch.query(f'Income > {min} & Income <= {i_q}')
    branch_incomes.append(i_branch)
    if len(branch) > 0:
      probability = len(i_branch) / len(branch)
      if probability > 0:
        entropies_incomes.append( -probability*math.log10(probability) )
len(entropies_incomes)

#FUENTE -> HOUSE
for home in house_own_types:
  h_branch = o_df.query(f'House_Ownership == "{home}"')
  branch_hown.append(h_branch)
  probability = len(h_branch) / total_instances
  if probability > 0:
    entropies_house.append( -probability*math.log10(probability) )
len(entropies_house)

#FUENTE -> PROFESSION
for prof in profession_types:
  p_branch = o_df.query(f'Profession == "{prof}"')
  branch_prof.append(p_branch)
  probability = len(p_branch) / total_instances
  if probability > 0:
    entropies_profession.append( -probability*math.log10(probability) )
len(entropies_profession)

f = open('/MINERIA_DATOS/LABORATORIO/Practica5/final_entropies.txt', 'a')
f.write(f'Entropy for Profession-House_Ownership-Age-Income: {category_entropies} H/S \n')
f.close()

entropies = [ 330.6761057516586  , 72.55327341825608  , 202.72933907528804  , 207.30876969097093  ,
             216.0038233777178  , 67.09499729442676  , 109.02204428517265  , 72.5434766579544  ,
              201.3213770336267  , 207.5627328773163  , 214.70882099417446  , 73.32764117260719  , 
              203.5037068296383  , 73.32050117298861  , 202.0984015486597  , 261.79888297393836  , 
              261.2543832057796  , 232.08978686747184  , 240.78484055421876  , 232.2854652025408  , 
              239.4315533193992  , 265.3580728116708  , 264.81357304351246 ]
min_entropy = sorted(entropies)[0]
