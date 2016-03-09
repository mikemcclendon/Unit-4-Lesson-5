import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans,kmeans2,vq

df = pd.read_csv('un.csv')

#printing out non-null values per column
#print(df.apply(pd.Series.nunique))

#filling na instead of dropping NaN to simplify calcs
df = df.fillna(0)
data = df[['lifeMale', 'lifeFemale', 'infantMortality', 'GDPperCapita']]

#calculating WCSS for the elbow. feel like missed the distance calculation
klist = range(1,11)
KM = [ kmeans(data, k) for k in klist ]
WCSS = [ v for (c,v) in KM ]

# #plotting elbow. should I have done this for each pair?
# plt.figure()
# plt.scatter(klist, WCSS)
# plt.xlabel('Number of clusters')
# plt.ylabel('Ave. WCSS')
# plt.title('Elbow Plot')
# plt.show()
# plt.clf()

#classifying the observations with KM2 and plotting
mortandGDP = df[['GDPperCapita', 'infantMortality']]
KM3 = kmeans2(mortandGDP, 3, minit='points')
plt.figure()
plt.scatter( df['GDPperCapita'], df['infantMortality'], c=KM3[1], cmap=plt.cm.Paired)
plt.xlabel('Per Capita GDP')
plt.ylabel('Infant Mortality/1000')
plt.show()
plt.clf()

