import pandas as pd
import matplotlib.pyplot as plt

def convert(val):
  try:
    return float(val)
  except:
    return None

df_surasia = pd.read_csv('/content/drive/MyDrive/FIME/NOVENO/MINERIA_DATOS/DATABANK/Dataset_surasiatico_2000.csv')

countries = df_surasia['Country Name'].unique()

current_country = df_surasia.query(f'`Country Name` == "{countries[8]}"')

gdp = current_country.query('`Series Code` == "NY.GDP.MKTP.PP.KD"')

ppp = current_country.query('`Series Code` == "PA.NUS.PPP"')

atms = current_country.query('`Series Code` == "FB.ATM.TOTL.P5"')

commercial = current_country.query('`Series Code` == "FB.CBK.BRWR.P3"')

borrowers = current_country.query('`Series Code` == "FB.CBK.BRCH.P5"')

ppp = ppp[['2010 [YR2010]','2011 [YR2011]','2012 [YR2012]','2013 [YR2013]','2014 [YR2014]','2015 [YR2015]','2016 [YR2016]','2017 [YR2017]','2018 [YR2018]','2019 [YR2019]','2020 [YR2020]']]
gdp = gdp[['2010 [YR2010]','2011 [YR2011]','2012 [YR2012]','2013 [YR2013]','2014 [YR2014]','2015 [YR2015]','2016 [YR2016]','2017 [YR2017]','2018 [YR2018]','2019 [YR2019]','2020 [YR2020]']]
atms = atms[['2010 [YR2010]','2011 [YR2011]','2012 [YR2012]','2013 [YR2013]','2014 [YR2014]','2015 [YR2015]','2016 [YR2016]','2017 [YR2017]','2018 [YR2018]','2019 [YR2019]','2020 [YR2020]']]
commercial = commercial[['2010 [YR2010]','2011 [YR2011]','2012 [YR2012]','2013 [YR2013]','2014 [YR2014]','2015 [YR2015]','2016 [YR2016]','2017 [YR2017]','2018 [YR2018]','2019 [YR2019]','2020 [YR2020]']]
borrowers = borrowers[['2010 [YR2010]','2011 [YR2011]','2012 [YR2012]','2013 [YR2013]','2014 [YR2014]','2015 [YR2015]','2016 [YR2016]','2017 [YR2017]','2018 [YR2018]','2019 [YR2019]','2020 [YR2020]']]

ppp_plot = list(map(lambda y: convert(ppp[y]),ppp))
gdp_plot = list(map(lambda y: convert(gdp[y]),gdp))
atms_plot = list(map(lambda y: convert(atms[y]),atms))
commer_plot = list(map(lambda y: convert(commercial[y]),commercial))
borrow_plot = list(map(lambda y: convert(borrowers[y]),borrowers))

fig, ax = plt.subplots()
ax.set_xlabel('GDP', fontsize=15)
ax.set_ylabel('PPP', fontsize=15)
ax.set_title('Poder adquisitivo aumenta si el acceso a la banca')
ax.scatter(gdp_plot,ppp_plot)

fig, ax = plt.subplots()
ax.set_xlabel('ATMS', fontsize=15)
ax.set_ylabel('PPP', fontsize=15)
ax.set_title('Poder adquisitivo aumenta si el acceso a la banca')
ax.scatter(atms_plot,ppp_plot)

fig, ax = plt.subplots()
ax.set_xlabel('Borrowers', fontsize=15)
ax.set_ylabel('PPP', fontsize=15)
ax.set_title('Poder adquisitivo aumenta si el acceso a la banca')
ax.scatter(borrow_plot,ppp_plot)

fig, ax = plt.subplots()
ax.set_xlabel('Sucursales', fontsize=15)
ax.set_ylabel('PPP', fontsize=15)
ax.set_title('Poder adquisitivo aumenta si el acceso a la banca')
ax.scatter(commer_plot,ppp_plot)
