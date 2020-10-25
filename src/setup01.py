import numpy as np
import pandas as pd
from pathlib import Path

from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, FactorRange

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
    dataf = dataf[dataf["COUNTRY_ALPHA"]==country]

    df_out = dataf.groupby(["COUNTRY_ALPHA", "S002", "education"])["immigrants"].count().to_frame().reset_index()
    df_specific = dataf[dataf["immigrants"]==1].groupby(["COUNTRY_ALPHA", "S002", "education"])["immigrants"].count().to_frame().reset_index()

    df_out["share"] = df_specific["immigrants"]/df_out["immigrants"]

    df_out = rightshape(df_out)
    return df_out

def rightshape(dataf):
    dataf = dataf.copy()

    dataf.sort_values(["S002", "education"], inplace=True)

    plot_dici = dict()
    plot_dici["years"] = [str(year) for year in np.unique(dataf["S002"])]

    for educ in np.unique(dataf["education"]).astype("int"):
        plot_dici[str(educ)] = dataf.loc[dataf["education"]==educ, "share"].tolist()


    return plot_dici

def plot_f1(dataf, country):
    dataf = dataf.copy()
    plot_dici = prepplot(dataf, country)

    years = plot_dici["years"]
    educ_levels = ["0", "1", "2"]
    labels = ["low", "middle", "high"]
    colors = ["#c9d9d3", "#718dbf", "#e84d60"]
    title = "Share of people positive towards immigrants by education level"

    p = figure(x_range=years, plot_height=250, title=title,
               toolbar_location=None, tools="")

    p.vbar_stack(educ_levels, x="years", width=0.3, color=colors,
                 source=plot_dici, legend_label=labels)

    show(p)

def get_data(dataf, country_list):
    dataf = dataf.copy()

    zero = []
    one = []
    two = []
    for country in country_list:
        dici = prepplot(dataf, country)

        zero = np.append(zero, dici["0"])
        one = np.append(one, dici["1"])
        two = np.append(two, dici["2"])

    return [zero, one, two]

def bar_data(dataf, country_list):
    dataf = dataf.copy()
    dataf.sort_values(["COUNTRY_ALPHA", "S002"], inplace=True)

    zero = []
    one = []
    two = []
    dataf = dataf[dataf["COUNTRY_ALPHA"].isin(country_list)]
    country_list = countries
    for country in country_list:
        df_relevant = dataf[dataf["COUNTRY_ALPHA"]==country]
        for w in np.sort(np.unique(df_relevant["S002"])):
            c_condition = df_relevant["COUNTRY_ALPHA"]==country
            w_condition = df_relevant["S002"]==w

            i_condition = df_relevant["immigrants"]==1
            mean = np.empty(3)
            for e in np.arange(3):
                e_condition = dataf["education"]==e
                mean[e] = sum(c_condition & w_condition & i_condition & e_condition) / sum(c_condition & w_condition)

            zero = np.append(zero, mean[0])
            one = np.append(one, mean[1])
            two = np.append(two, mean[2])
    return [zero, one, two]

def make_p1(factors, values):
    src = ColumnDataSource(data=dict(
        x = factors,
        zero = np.ravel(values[0]),
        one = np.ravel(values[1]),
        two = np.ravel(values[2])
    ))

    title1 = "Share of people positive towards immigrants by education level"
    educ_levels = ["zero", "one", "two"]
    labels = ["low", "middle", "high"]
    colors = ["#c9d9d3", "#718dbf", "#e84d60"]

    p1 = figure(x_range=FactorRange(*factors), plot_height=450,
               toolbar_location=None, tools="", title=title1)

    p1.vbar_stack(educ_levels, x="x", width=0.9, alpha=0.85, color=colors, source=src,
                 legend_label=labels)

    p1 = make_pretty(p1)
    return p1

def make_p2(p1, factors, values):
    title2 = "Mean of people positive towards immigrants by education level"
    educ_levels = ["zero", "one", "two"]
    labels = ["low", "middle", "high"]
    colors = ["#c9d9d3", "#718dbf", "#e84d60"]

    p2 = figure(x_range=p1.x_range, y_range=p1.y_range, plot_height=450,
               toolbar_location=None, tools="", title=title2)

    for i in np.arange(3):
        p2.circle(x=factors, y = values[i], fill_color = colors[i], size=8)

    p2 = make_pretty(p2)
    return p2

def make_pretty(p):
    p.xgrid.grid_line_color = None
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"

    return p

def plot_countries(dataf, country_list):
    dataf = dataf.copy()

    dataf.sort_values(["COUNTRY_ALPHA", "S002"], inplace=True)
    country_list.sort()

    # Only selecting relevant countries
    df_relevant = dataf[dataf["COUNTRY_ALPHA"].isin(country_list)]

    # Only taking those where we have data on immigration and education
    nomissing = (df_relevant["immigrants"].isna()==False)&(df_relevant["education"].isna()==False)
    df_relevant = df_relevant[nomissing]

    # Grouping them for plot
    small_df = df_relevant.groupby(["COUNTRY_ALPHA", "S002"]).size().to_frame().reset_index()
    x = list(zip(small_df["COUNTRY_ALPHA"], small_df["S002"].astype("str")))

    # Getting data for the bars
    values = bar_data(df_relevant, country_list)
    p1 = make_p1(x, values)

    # Getting data for the circles
    values_circles = get_data(df_relevant, country_list)
    p2 = make_p2(p1, x, values_circles)

    p = gridplot([[p1, p2]], toolbar_location=None)
    show(p)


df = make_df1()
df1 = recode(df)

# Analysis
df1[df1["S020"]!= 1998].groupby("COUNTRY_ALPHA")["trade_dummy"].mean()

# Trying weights
df1["weighted_td"] = df1["S017"] * df1["trade_dummy"]
df1.groupby("COUNTRY_ALPHA")["weighted_td"].mean()

plot_f1(df1, "AUS")

countries = ["DEU", "AUS", "USA", "HUN", "JPN"]
plot_countries(df1, countries)


## Q2
# Reading in data
df = pd.read_csv(wid_path / "WID_Data_22102020-165529.csv", sep=";", skiprows=1)
df.dropna()
