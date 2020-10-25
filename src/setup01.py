import numpy as np
import pandas as pd
from pathlib import Path

from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show

dir = Path.cwd()
data_path = Path.cwd() / "data/"
wid_path = data_path / "wid_data/"

## Q1
# Reading in data
def make_df1():
    itr = pd.read_stata(data_path / "WVS_TimeSeries_stata_v1_2.dta", iterator=True)

    df = itr.read(convert_categoricals=False)
    labels = itr.variable_labels()
    # Including relevant variables such as year, country
    base_vars = [1, 2, 3, 16, 17, 18, 19]
    df_base = df.iloc[:, base_vars].copy()
    #Adding variable of interest
    interest_vars = ["E062", "A124_02", "A124_06", "A124_16", "X025", "X025A2", "X025R"]
    df_out = pd.concat([df_base, df[interest_vars]], axis=1)

    return df_out

# Recoding variables for our use
def recode(dataf):
    dataf = dataf.copy()

    # Coding trade dummy
    dataf["trade_dummy"] = np.nan
    dataf.loc[dataf["E062"]==1, "trade_dummy"]=1
    dataf.loc[dataf["E062"]==2, "trade_dummy"]=0
    dataf.loc[dataf["E062"]==-1, "trade_dummy"]=0

    # Coding education
    dataf["education"] = np.nan
    dataf.loc[(dataf["X025"]==1)|(dataf["X025"]==2)|(dataf["X025"]==2),
              "education"] = 0
    dataf.loc[(dataf["X025"]==4)|(dataf["X025"]==5)|(dataf["X025"]==6),
              "education"] = 1
    dataf.loc[(dataf["X025"]==7)|(dataf["X025"]==8),
              "education"] = 2

    # Renaming neighbours
    dataf.rename(columns={"A124_02": "race",
                          "A124_06": "immigrants"},
                 inplace=True)
    return dataf

def prepplot(dataf, country):
    dataf = dataf.copy()
    df_out = dataf.groupby(["COUNTRY_ALPHA",
                            "S020", "education"])["immigrants"].mean().to_frame().reset_index()

    return df_out[df_out["COUNTRY_ALPHA"]==country]

def plot_f1(dataf, country):
    dataf = dataf.copy()
    dataf = prepplot(dataf, country)

    p1 = figure(title="US immigration attitudes by education")

    p1.line(dataf.loc[dataf["education"]==0, "S020"],
            dataf.loc[dataf["education"]==0, "immigrants"],
            line_color="orange",
            legend_label="low education")

    p1.line(dataf.loc[dataf["education"]==1, "S020"],
            dataf.loc[dataf["education"]==1, "immigrants"],
            line_color="blue",
            legend_label="middle education")

    p1.line(dataf.loc[dataf["education"]==2, "S020"],
            dataf.loc[dataf["education"]==2, "immigrants"],
            line_color="green",
            legend_label="high education")

    show(p1)

df = make_df1()
df1 = recode(df)

# Analysis
df1[df1["S020"]!= 1998].groupby("COUNTRY_ALPHA")["trade_dummy"].mean()

# Trying weights
df1["weighted_td"] = df1["S017"] * df1["trade_dummy"]
df1.groupby("COUNTRY_ALPHA")["weighted_td"].mean()

plot_f1(df1, "USA")


## Q2
# Reading in data
df = pd.read_csv(wid_path / "WID_Data_22102020-165529.csv", sep=";", skiprows=1)
df.dropna()
