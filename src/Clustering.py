#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

notes = pd.read_csv('data/processed_notes.csv')
notes


# ### Notes:
# 
# - We know that there are 10 patients and 42146 documents. These 10 patients themselves might be clusters. 
# 
# - So, remove the 'case_num' column to make the problem completely Unsupervised learning.
# 
# - Will have to consider the inclusion of processed_note_length and initial_note_length later.

# In[8]:


# setting X and y for modelling

X = notes[['processed_notes', 'initial_note_length', 'processed_note_length']]
y = notes['case_num']
 


# Initialize the vectorizer with 1-grams and other hyperparameters as needed
# Since the preprocessing is already done, we don't need to specify a custom tokenizer or stop words
vectorizer = TfidfVectorizer(ngram_range=(1,1), 
                             min_df = 0.001, 
                             max_df = 0.9)

# Fit and transform th e preprocessed 'pn_history' column to create the DTM
dtm = vectorizer.fit_transform(X['processed_notes'])
dtm_dense = dtm.todense()

# Convert DTM to a DataFrame
dtm_df = pd.DataFrame(dtm_dense, columns=vectorizer.get_feature_names_out())



# Describe the DTM
print("Size of the DTM: ", dtm_df.shape) 
memory_usage = dtm_df.memory_usage(deep=True).sum()
print("Memory usage (in bytes): ", memory_usage)
dtm_df


# ### Notes:
# 
# - As predicted in the EDA, the words like year, past, history cannot be found in the Document Term Matrix.
# 
# 

# In[9]:


def cluster_score_plots(min_cluster, max_cluster, inertia_values, silhouette_scores, save_name):

    sns.set(style="darkgrid")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    # Plot the inertia values
    axes[0].plot(range(min_cluster,max_cluster+1), inertia_values, marker='o') 
    axes[0].set_title('Inertia')
    axes[0].set_xlabel('Number of clusters')
    axes[0].set_ylabel('Inertia')  


    # Plot the silhouette scores
    axes[1].plot(range(min_cluster,max_cluster+1), silhouette_scores, marker='o') 
    axes[1].set_title('Silhouette Score') 
    axes[1].set_xlabel('Number of clusters') 
    axes[1].set_ylabel('Silhouette Score')
    plt.savefig(save_name)
    plt.show()



# In[10]:


min_cluster = 3
max_cluster = 14


# ## Dimensionality Reduction

# ### i. Using TruncatedSVD

# In[11]:


from sklearn.decomposition import TruncatedSVD

lsa = TruncatedSVD(n_components=2)
X_lsa = lsa.fit_transform(dtm)

explained_variance = lsa.explained_variance_ratio_.sum()

print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")


# In[12]:


lsa_df = pd.DataFrame(data = X_lsa, columns = ['lsa1', 'lsa2'])

plt.scatter(lsa_df['lsa1'],lsa_df['lsa2'])
plt.xlabel('lsa1')
plt.ylabel('lsa2')
plt.title('TruncatedSVD (2 components)')
plt.savefig('figs/2_component_lsa.png')
plt.show()


# ### Notes:
# 
# - The 2 components have a cumulative explained variance score of 3.9% only.
# 
# - This means that we lost 96% of the original variance. 
# 
# - Lets see how the k-means clustering goes.

# In[13]:


from sklearn.decomposition import TruncatedSVD

lsa = TruncatedSVD(n_components=1000)
X_lsa = lsa.fit_transform(dtm)
explained_variance = lsa.explained_variance_ratio_.sum()
print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")


inertia_values = []
silhouette_scores = []

for num_clust in range(min_cluster,max_cluster+1):
    kmeans = KMeans(n_clusters=num_clust, random_state=42, n_init='auto').fit(X_lsa)
    cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
    print(f"Number elements in {num_clust} clusters: {cluster_sizes}")
    
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_lsa, kmeans.labels_))


cluster_score_plots(min_cluster, max_cluster, inertia_values, silhouette_scores, 'figs/lsa_kmeans_scores.png')

print('Best silhouette score for TruncatedSVD 1000 components:', max(silhouette_scores))
print(f'- Best silhouette score is observed at n_clusters = {min_cluster + silhouette_scores.index(max(silhouette_scores))}')

print(f'- n_clusters account to Inertia scores are:', inertia_values.index(KneeLocator(inertia_values, list(range(min_cluster, max_cluster+1)), curve='convex', direction='decreasing').knee))


# ### ii. Using UMAP

# In[52]:


reducer = umap.UMAP(n_neighbors = 5, n_components = 2, metric = 'euclidean', min_dist = 0.05, spread = 1.0, random_state=42)
projected_data = reducer.fit_transform(dtm)

red1 = pd.DataFrame(data=projected_data, columns=['umap_1','umap_2'])
display(red1)

plt.scatter(red1['umap_1'],red1['umap_2'], alpha = 0.3)
plt.title('UMAP 2-components')
plt.xlabel('umap_1')
plt.ylabel('umap_2')
plt.savefig('figs/umap_2_component.png')
plt.show()


# - When UMAP is used for dimensionality reduction (2 components), those components have formed 11 noticable clusters.
# 
# 

# #### UMAP K-means

# In[53]:


inertia_values = []
silhouette_scores = []

for num_clust in range(min_cluster,max_cluster+1):
    kmeans = KMeans(n_clusters=num_clust, random_state=42, n_init='auto').fit(projected_data)
    cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
    print(f"Number elements in {num_clust} clusters: {cluster_sizes}")
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(projected_data, kmeans.labels_))

cluster_score_plots(min_cluster, max_cluster, inertia_values, silhouette_scores, 'figs/umap_kmeans_scores.png')

print('Best silhouette score for UMAP 2 components:', max(silhouette_scores))
print(f'- Best silhouette score is observed at n_clusters = {min_cluster + silhouette_scores.index(max(silhouette_scores))}')

print(f'- n_clusters account to Inertia scores are:', inertia_values.index(KneeLocator(inertia_values, list(range(min_cluster, max_cluster+1)), curve='convex', direction='decreasing').knee))


# - The inertia scores don't show a clear elbow to decide number of clusters.
# 
# - The silhouette scores indicate that the number of clusters = 10.

# In[54]:


#plotting for kmeans 10 clusters and Checking against the target- case_num

kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto').fit(projected_data)

sns.set(style="darkgrid")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

axes[0].scatter(red1['umap_1'],red1['umap_2'], c=kmeans.labels_, cmap='Spectral')
axes[0].set_title('UMAP 2-component - K-means')
axes[0].set_xlabel('umap_1')
axes[0].set_ylabel('umap_2')


axes[1].scatter(red1['umap_1'],red1['umap_2'], c=y, cmap='Spectral')
axes[1].set_title('UMAP 2-component - with case_num (target)')
axes[1].set_xlabel('umap_1')
axes[1].set_ylabel('umap_2')

plt.savefig('figs/umap_kmeans_casenum.png')
plt.show()


# In[55]:


from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix

mapping = {6: 0, 
           9: 1, 
           4: 2, 
           1: 3, 
           0: 4, 
           3: 5, 
           8: 6, 
           5: 7, 
           7: 8, 
           2: 9}

relabeled_labels = np.vectorize(mapping.get)(kmeans.labels_)
print('Adjusted rand score for the Kmeans clustering solution:',adjusted_rand_score(y, relabeled_labels))
print('Contingency matrix:')
contingency_matrix(y, relabeled_labels)


# - We can see that there is overlap in patient case_num from the clusters formed.
# 
# - Considering the context of patient notes, this overlap might be due to the similar symptoms.

# #### UMAP HDBSCAN

# In[56]:


from sklearn.cluster import HDBSCAN
import seaborn as sns

hdb_tsne = HDBSCAN(min_cluster_size=500, min_samples = 100)
cluster_labels = hdb_tsne.fit_predict(red1)

plt.figure(figsize = (20,15))

palette = sns.color_palette('dark', np.unique(cluster_labels).max() + 1)
palette[7] = (0.8, 0.1, 0.1)
palette[5] = (0.1, 0.3, 0.7)
palette[3] = (0.5, 0.8, 0)
palette[0] = (0, 0.8, 0.8)

#plotting clusters
for i in range(np.unique(cluster_labels).max() + 1):
    plt.scatter(red1[cluster_labels == i]['umap_1'], red1[cluster_labels == i]['umap_2'], color = palette[i], label = f'Cluster {i+1}')

#plotting noise
plt.scatter(red1[cluster_labels == -1]['umap_1'], red1[cluster_labels == -1]['umap_2'], color = 'black', marker = 'x', label = f'Noise')


plt.title('HDBSCAN Clustering (2D UMAP)')
plt.xlabel('umap1')
plt.ylabel('umap2')
plt.xlim(-10, 30)
plt.ylim(-10, 25)
plt.legend(loc = 'best', fontsize = 'medium', ncols =2)
plt.savefig('figs/hdbscan_clusters.png')
plt.show()


# In[65]:


clustering_df = pd.DataFrame()
clustering_df['y'] = y
clustering_df['y_pred'] = cluster_labels
no_noise = clustering_df[clustering_df['y_pred'] != -1]

mapping = {6: 0, 
           0: 1, 
           3: 2, 
           2: 3, 
           7: 4, 
           9: 5, 
           5: 6, 
           8: 7, 
           1: 8, 
           4: 9,
          10:10}

no_noise['y_pred'] = no_noise['y_pred'].replace(mapping)


print('Adjusted rand score for the DBSCAN clustering solution:',adjusted_rand_score(no_noise['y'], no_noise['y_pred']))
print('Contingency matrix:')
contingency_matrix(no_noise['y'], no_noise['y_pred'])


# ### Using T-SNE

# In[66]:


tsne = TSNE(n_components=2,  random_state = 42, init='random')
tsne_result = tsne.fit_transform(dtm)
red2 = pd.DataFrame(data=tsne_result, columns=['tsne1','tsne2'])
red2


plt.scatter(red2['tsne1'],red2['tsne2'])
plt.title('T-SNE 2-components')
plt.xlabel('tsne_1')
plt.ylabel('tsne_2')
plt.savefig('figs/tsne_2components.png')
plt.show()


# #### T-SNE KMeans

# In[67]:


inertia_values = []
silhouette_scores = []

for num_clust in range(min_cluster,max_cluster+1):
    kmeans = KMeans(n_clusters=num_clust, random_state=42, n_init='auto').fit(tsne_result)
    cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
    print(f"Number elements in {num_clust} clusters: {cluster_sizes}")
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(tsne_result, kmeans.labels_))

cluster_score_plots(min_cluster, max_cluster, inertia_values, silhouette_scores, 'figs/tsne_kmeans_scores.png')

print('Best silhouette score for T-SNE 2 components:', max(silhouette_scores))
print(f'- Best silhouette score is observed at n_clusters = {min_cluster + silhouette_scores.index(max(silhouette_scores))}')

print(f'- n_clusters account to Inertia scores are:', inertia_values.index(KneeLocator(inertia_values, list(range(min_cluster, max_cluster+1)), curve='convex', direction='decreasing').knee))


# In[68]:


#plotting for kmeans 10 clusters and Checking against the target- case_num

kmeans = KMeans(n_clusters=10, random_state=42, n_init='auto').fit(tsne_result)

sns.set(style="darkgrid")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

axes[0].scatter(red2['tsne1'],red2['tsne2'], c=kmeans.labels_, cmap='Spectral')
axes[0].set_title('T-SNE 2-component -KMeans')
axes[0].set_xlabel('tsne_1')
axes[0].set_ylabel('tsne_2')


axes[1].scatter(red2['tsne1'],red2['tsne2'], c=y, cmap='Spectral')
axes[1].set_title('T-SNE 2-component - with case_num (target)')
axes[1].set_xlabel('tsne_1')
axes[1].set_ylabel('tsne_2')
plt.savefig('figs/tnse_kmeans_casenum.png')
plt.show()


# In[69]:


'''mapping = {6: 0, 
           9: 1, 
           4: 2, 
           1: 3, 
           0: 4, 
           3: 5, 
           8: 6, 
           5: 7, 
           7: 8, 
           2: 9}

relabeled_labels = np.vectorize(mapping.get)(kmeans.labels_)'''
print('Adjusted rand score for the Kmeans clustering solution:',adjusted_rand_score(y, kmeans.labels_))
print('Contingency matrix:')
contingency_matrix(y, kmeans.labels_)


# #### T-SNE HDBSCAN

# In[70]:


from sklearn.cluster import HDBSCAN
import seaborn as sns

hdb_tsne = HDBSCAN(min_cluster_size=500, min_samples = 100)
cluster_labels = hdb_tsne.fit_predict(red2)

plt.figure(figsize = (20,15))

palette = sns.color_palette('dark', np.unique(cluster_labels).max() + 1)
palette[7] = (0.8, 0.1, 0.1)
palette[5] = (0.1, 0.3, 0.7)
palette[3] = (0.5, 0.8, 0)
palette[0] = (0, 0.8, 0.8)

#plotting clusters
for i in range(np.unique(cluster_labels).max() + 1):
    plt.scatter(red2[cluster_labels == i]['tsne1'], red2[cluster_labels == i]['tsne2'], color = palette[i], label = f'Cluster {i+1}')

#plotting noise
plt.scatter(red2[cluster_labels == -1]['tsne1'], red2[cluster_labels == -1]['tsne2'], color = 'black', marker = 'x', label = f'Noise')


plt.title('HDBSCAN Clustering (2D T-SNE)')
plt.xlabel('tsne1')
plt.ylabel('tsne2')
plt.xlim(-80, 80)
plt.ylim(-80, 80)
plt.legend(loc = 'best', fontsize = 'medium', ncols =2)
plt.savefig('figs/hdbscan_clusters.png')
plt.show()


# In[72]:


clustering_df = pd.DataFrame()
clustering_df['y'] = y
clustering_df['y_pred'] = cluster_labels
no_noise = clustering_df[clustering_df['y_pred'] != -1]

mapping = {4: 0, 
           1: 1, 
           3: 2, 
           7: 3, 
           9: 4, 
           8: 5, 
           0: 6, 
           5: 7, 
           2: 8, 
           6: 9}

no_noise['y_pred'] = no_noise['y_pred'].replace(mapping)


print('Adjusted rand score for the DBSCAN clustering solution:',adjusted_rand_score(no_noise['y'], no_noise['y_pred']))
print('Contingency matrix:')
contingency_matrix(no_noise['y'], no_noise['y_pred'])


# In[ ]:




