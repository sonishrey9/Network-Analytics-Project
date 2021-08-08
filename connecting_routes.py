# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import networkx as nx 
import matplotlib.pyplot as plt


# %%
# Degree Centrality
G = pd.read_csv("connecting_routes.csv", names = ["flights", " ID", "main Airport", "main Airport ID", "Destination","Destination  ID", "0","haults","machinary"] )
G


# %%
G = G.drop(['0'], axis = 1)
G


# %%
## Here we will check the percentage of nan values present in each feature
## 1 -step make the list of features which has missing values
features_with_na=[features for features in G.columns if G[features].isnull().sum()>1]
## 2- step print the feature name and the percentage of missing values

for feature in features_with_na:
    print(feature, np.round(G[feature].isnull().mean() * 100 , 4),  ' % missing values')


# %%
# Dropping all na  values as Data set is Big

G.dropna(inplace=True)


# %%
G = G.iloc[:, 1:10]


# %%
G


# %%
g = nx.Graph() #Create an empty graph with no nodes and no edges.


# %%
G.info()


# %%
g = nx.from_pandas_edgelist(G, source = 'main Airport', target = 'Destination')


# %%

print(nx.info(g))


# %%
b = nx.degree_centrality(g)  # Degree Centrality
print(b)

### Degree centrality is the simplest centrality measure to compute. Recall that a node's degree is simply a count of how many social connections (i.e., edges) it has. The degree centrality for a node is simply its degree. A node with 10 social connections would have a degree centrality of 10. A node with 1 edge would have a degree centrality of 1.


# %%

# pos = nx.spring_layout(g, k = 0.15)
# nx.draw_networkx(g, pos, node_size = 10, node_color = 'blue')


# %%
# ## Betweeness Centrality 
# b = nx.betweenness_centrality(g) # Betweeness_Centrality
# print(b)


# %%
nx.pagerank(g, max_iter=600) 


# %%

## Eigen-Vector Centrality
evg = nx.eigenvector_centrality(g, max_iter=600) # Eigen vector centrality
print(evg)


# %%
# Eigen_df = pd.DataFrame.from_dict(evg, orient='index').T
Eigen_df  = pd.DataFrame(evg, index=[1]) 
Eigen_df


# %%
Eigen_df_transpose = Eigen_df.transpose()
Eigen_df_transpose


# %%
# cluster coefficient
cluster_coeff = nx.clustering(g)
print(cluster_coeff)


# %%
cluster_coeff_df = pd.DataFrame.from_dict(cluster_coeff, orient='index').T
cluster_coeff_df


# %%
# Average clustering
cc = nx.average_clustering(g) 
print(cc)


# %%
from matplotlib.pyplot import figure
figure(figsize=(10, 8))
nx.draw_shell(g)


# %%



