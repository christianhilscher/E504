import numpy as np 
import pandas as pd 
import os 
from bokeh.plotting import figure, output_file, show

wd_lc = "/Users/llccf/OneDrive/Dokumente/3. Semester/International Trade and Tax Policy/Problem Sets/PS2/"
os.chdir(wd_lc)

#*##############################
#! Q1 b) - Mayda and Rodrik (2005) Table 2
#*##############################

#! Data 
#world value survey wave 1995 - 1998 
wvs = pd.read_stata("data/WVS1994-1998.dta", convert_categoricals=False)

#keep only relevant countries 
#for country codes see documentation
countries = ['Uruguay', 'Venezuela', 'Brazil', 'Argentina', 'Mexico', 'Peru', 'India', 'Turkey', 
                'Bangladesh', 'Pakistan', 'Puerto Rico', 'Chile', 'Poland', 'Australia', 
                'China', 'Dom Rep', 'Spain', 'USA', 'Philippines', 'Lithuania', 'Slovenia', 'Russia', 
                'Moldova', 'S. Africa', 'S. Korea', 'Bulgaria', 'Switzerland',  'Germany', 'Macedonia',
                'Finland', 'Sweden', 'Estonia', 'Taiwan', 'Latvia', 'Croatia', 'Norway', 'Bosnia', 
                'Nigeria', 'Serbia', 'Armenia', 'Ukraine', 'Azerbaijan', 'Belarus', 'Georgia', 
                'Montenegro', 'Japan']
c_codes = [858, 862, 76, 32, 484, 604, 356, 792, 50, 586, 630, 152, 616, 36, 156, 
            42, 214, 724, 840, 608, 440, 705, 643, 498, 710, 410, 100, 756, 276, 807, 
            246, 752, 233, 158, 191, 578, 914, 566, 911, 51, 804, 31, 112, 268, 912, 392]

c_dict = {i: j for i, j in zip(c_codes, countries)}
#Mayda and Rodrik use data from 1995 - 1997
year = [1995, 1996, 1997]

#now drop everythin that we are not interested in 
wvs_oi = wvs[wvs['V238'].isin(year)].reset_index(drop = True)
#wvs_oi = wvs_oi[wvs_oi['V2'].isin(c_codes)].reset_index(drop = True)
#also, we only need one variable, keep country identifier and this variable 
var_oi = wvs_oi[['V2', 'V133']]
#now, need to create a new variable of this, i.e. their identifier for pro- and anti-trade 
var_oi['protrade'] = 1
var_oi['protrade'][var_oi['V133'] == 2] = 0 
var_oi['protrade'][var_oi['V133'] == -1] = 0 
#! Create average by country 
table = var_oi.groupby('V2').mean().reset_index().drop('V133', axis = 1)
table['countries'] = table.set_index('V2').index.map(c_dict.get)
table = table.sort_values('protrade')
print(table)


#*##############################
#! Q2 b) - Alvaredo et al. (2013) Figure 2 
#*##############################

income = pd.read_csv('data/Alvaredoetal_Fig2_countries.csv', sep = ';').reset_index()
income.columns = income.iloc[0, :]
income = income.drop(0)