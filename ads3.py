# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:47:46 2023

@author: DEVI
"""

import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import cluster_tools as ct
import importlib
import scipy.optimize as opt

world_data = pd.read_csv("API_AG.LND.AGRI.ZS_DS2_en_csv_v2_5447782.csv",\
                         skiprows = 4)
print(world_data)

print(world_data.describe())
print()

df_data = world_data[["1961", "1971", "1981", "1991", "2001", "2011",'2020']]
print(df_data.describe())


corr_matrix = df_data.corr()
print(corr_matrix)

    
pd.plotting.scatter_matrix(df_data, figsize=(12, 12), s=5, alpha=0.8)

plt.show()

df_data1 = df_data[["1991", "2011"]]  # extract the two columns for clustering


df_data1 = df_data1.dropna()  # entries with one nan are useless
df_data1 = df_data1.reset_index()
print(df_data1.iloc[0:15])

# reset_index() moved the old index into column index

# remove before clustering
df_data1 = df_data1.drop("index", axis=1)
print(df_data1.iloc[0:15])

df_norm, df_min, df_max = ct.scaler(df_data1)
print()
print("n  score")

# loop over number of clusters
for ncluster in range(2, 10):
    
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_norm)     # fit done on x,y pairs

    labels = kmeans.labels_
    
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_

    # calculate the silhoutte score
    print(ncluster, skmet.silhouette_score(df_data1, labels))


ncluster = 4 # best number of clusters

# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)

# Fit the data, results are stored in the kmeans object
kmeans.fit(df_norm)     # fit done on x,y pairs

labels = kmeans.labels_
    
# extract the estimated cluster centres
cen = kmeans.cluster_centers_

cen = np.array(cen)
xcen = cen[:, 0]
ycen = cen[:, 1]


# cluster by cluster
plt.figure(figsize=(8.0, 8.0))

cm = plt.cm.get_cmap('tab10')
plt.scatter(df_norm["1991"], df_norm["2011"], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("area of ag_lnd(1991)")
plt.ylabel("area of ag_lnd(2011)")
plt.show()

print(cen)


# Applying the backscale function to convert the cluster centre
scen = ct.backscale(cen, df_min, df_max)
print()
print(scen)

xcen = scen[:, 0]
ycen = scen[:, 1]

# cluster by cluster
plt.figure(figsize=(8.0, 8.0))

cm = plt.cm.get_cmap('tab10')
plt.scatter(df_data1["1991"], df_data1["2011"], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("area of ag_lnd(1990)")
plt.ylabel("area of ag_lnd(2011)")
plt.show() 



df_ag_lnd = pd.read_csv("API_AG.LND.AGRI.ZS_DS2_en_csv_v2_5447782.csv",skiprows=4)
df_ag_lnd=df_ag_lnd.set_index('Country Name', drop=True)
df_ag_lnd=df_ag_lnd.loc[:,'1961':'2020']
df_ag_lnd=df_ag_lnd.transpose()
df_ag_lnd=df_ag_lnd.loc[:,'American Samoa']
df=df_ag_lnd.dropna(axis=0)
print(df.values)

df_ag_lnd=pd.DataFrame()

df_ag_lnd['Year']=pd.DataFrame(df.index)
df_ag_lnd['country Name']=pd.DataFrame(df.values)

print(df_ag_lnd.head())

df_ag_lnd.plot("Year", "country Name")
plt.show()

def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    t = t - 1981
    f = n0 * np.exp(g*t)
    return f

print(type(df_ag_lnd["Year"].iloc[1]))
df_ag_lnd["Year"] = pd.to_numeric(df_ag_lnd["Year"])
print(type(df_ag_lnd["Year"].iloc[1]))
param, covar = opt.curve_fit(exponential, df_ag_lnd["Year"],\
                             df_ag_lnd["country Name"], p0=(1.2e12, 0.03))

print("df_ag_lnd 1981", param[0]/1e9)
print("growth rate", param[1])

plt.figure()
plt.plot(df_ag_lnd["Year"], exponential(df_ag_lnd["Year"],\
                                        1.2e12, 0.03), label="trial fit")
plt.xlabel("Year")
plt.legend()
plt.show()


df_ag_lnd["fit"] = exponential(df_ag_lnd["Year"], *param)

df_ag_lnd.plot("Year", ["country Name", "fit"])
plt.show()


def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    
    f = n0 / (1 + np.exp(-g*(t - t0)))
    
    return f

importlib.reload(opt)

param, covar = opt.curve_fit(logistic, df_ag_lnd["Year"],
                             df_ag_lnd["country Name"], 
                              p0=(1.2e12, 0.03, 1981.0),maxfev=5000)


sigma = np.sqrt(np.diag(covar))

df_ag_lnd["fit"] = logistic(df_ag_lnd["Year"], *param)

df_ag_lnd.plot("Year", ["country Name", "fit"])
plt.show()

print("turning point", param[2], "+/-", sigma[2])
print("% of land at turning point", param[0]/1e9, "+/-", sigma[0]/1e9)
print("growth rate", param[1], "+/-", sigma[1])

df_ag_lnd["trial"] = logistic(df_ag_lnd["Year"], 3e12, 0.10, 1981)

df_ag_lnd.plot("Year", ["country Name", "trial"])
plt.show()

year = np.arange(1961, 2011)
forecast = logistic(year, *param)

plt.figure()
plt.plot(df_ag_lnd["Year"], df_ag_lnd["country Name"], label="country Name")
plt.plot(year, forecast, label="forecast")

plt.xlabel("year")
plt.ylabel("% of agricultural land")
plt.legend()
plt.show()


import errors as err

low, up = err.err_ranges(year, logistic, param, sigma)

plt.figure()
plt.plot(df_ag_lnd["Year"], df_ag_lnd["country Name"],\
         label="% of agricultural land")
plt.plot(year, forecast, label="forecast")

plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("% of agricultural land")
plt.legend()
plt.show()

print(logistic(2030, *param)/1e9)
print(err.err_ranges(2030, logistic, param, sigma))

# assuming symmetrie estimate sigma
gdp2030 = logistic(2030, *param)/1e9

low, up = err.err_ranges(2030, logistic, param, sigma)
sig = np.abs(up-low)/(2.0 * 1e9)
print()
print("% of agricultural land 2030", "% of agricultural land2030", "+/-", sig)




