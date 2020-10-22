import numpy as np
import pandas as pd
from pathlib import Path


dir = Path.cwd()
data_path = Path.cwd() / 'data/'


itr = pd.read_stata(data_path / 'WVS_TimeSeries_stata_v1_2.dta', iterator=True)

df = itr.read(convert_categoricals=False)
labels = itr.variable_labels()


# Including relevant variables such as year, country
base_vars = [1, 2, 3, 16, 17, 18, 19]
df_base = df.iloc[:, base_vars].copy()
#Adding variable of interest
df_out = pd.concat([df_base, df["E062"]], axis=1)

# Coding trade dummy
df_out["trade_dummy"] = np.nan
df_out.loc[df_out["E062"]==1, "trade_dummy"]=1
df_out.loc[df_out["E062"]==2, "trade_dummy"]=0
df_out.loc[df_out["E062"]==-1, "trade_dummy"]=0

# Analysis
df_out.groupby("COUNTRY_ALPHA")["trade_dummy"].mean()

# Trying weights
df_out["weighted_td"] = df_out["S017"] * df_out["trade_dummy"]
df_out[df_out["S002"]==3].groupby("COUNTRY_ALPHA")["weighted_td"].mean()
