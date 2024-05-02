'''Word2Vec on IsoSig data'''
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import string
from gensim.models import Word2Vec

#Define the alphabet for onehot encoding
alphabet = list(string.ascii_lowercase) + list(string.ascii_uppercase) + [str(i) for i in range(10)] + ['+','-']

mflds_names = ['S3','S2xS1','RP3','L71','L72','T3','PHS','H_SC']
mflds_strnames = [r'$S^3$',r'$S^2 \times S^1$',r'$\mathbb{RP}^3$',r'$L(7,1)$',r'$L(7,2)$',r'$T^3$',r'$PHS$',r'$H_{SC}$']
subfolder = 'MinimisingPaths'
moves = '23'

#%% #Import data
def importdataSL(mfld_idx):
    paths = []
    with open('../Data/Databases/'+subfolder+'/'+subfolder+f'_{mflds_names[mfld_idx]}.txt','r') as file:
        for line in file.readlines():
            paths.append(eval(line))
    return paths 

#Import the respective datasets
minimising_paths = []
for mfld_idx in range(len(mflds_names)):
    paths = importdataSL(mfld_idx)
    minimising_paths.append(paths)
print(f'Dataset lengths: {list(map(len,minimising_paths))}')
del(mfld_idx,paths)

#Identify path endpoints
endpoints = []
for manifold in minimising_paths:
    endpoints.append([])
    for path in manifold:
        if path[-1] not in  endpoints[-1]:
            endpoints[-1].append(path[-1])
del(manifold,path)  

#Define function to return the number of tetrahedra
def numtet(sig):
    return alphabet.index(sig[0])


#%% #Define and train the model
w2v_2   = Word2Vec(sentences=[path for paths in minimising_paths for path in paths], vector_size=2,   min_count=1) #vector_size = embedding vector size, min_count = ignore words with < this frequency
w2v_100 = Word2Vec(sentences=[path for paths in minimising_paths for path in paths], vector_size=100, min_count=1)

#%% #Make manifold isosig lists
manifolds = [np.sort(np.unique([sig for path in manifold for sig in path])) for manifold in minimising_paths]
embeddings_2   = [np.array([w2v_2.wv[sig]   for sig in manifold]) for manifold in manifolds]
embeddings_100 = [np.array([w2v_100.wv[sig] for sig in manifold]) for manifold in manifolds]
tet_counts = [[numtet(sig) for sig in manifold] for manifold in manifolds]

#%% #2d heatmap - manifolds
plt.figure()
for idx in range(len(embeddings_2)):
    plt.scatter(embeddings_2[idx][:,0],embeddings_2[idx][:,1],label=mflds_strnames[idx])
plt.legend(loc='upper left',bbox_to_anchor=(1., 1.))
plt.grid()
plt.tight_layout()
#plt.savefig('./isosig_2dheatmap_manifolds.png')

#%% #Resort data according to the number of tet 
min_numtet, max_numtet = min(map(min,tet_counts)), max(map(max,tet_counts))
embeddings_tetsort = [[] for i in range(min_numtet,max_numtet+1)]
for m_idx, manifold in enumerate(embeddings_2):
    for s_idx, sig in enumerate(manifold):
        embeddings_tetsort[tet_counts[m_idx][s_idx]-min_numtet].append(sig)
embeddings_tetsort = [np.array(i) for i in embeddings_tetsort]

#%% #2d heatmap - number of tetrahedra
plt.figure()
for idx in range(len(embeddings_tetsort)):
    plt.scatter(embeddings_tetsort[idx][:,0],embeddings_tetsort[idx][:,1],label=str(min_numtet+idx))
plt.legend(loc='upper left',bbox_to_anchor=(1., 1.02))
plt.grid()
plt.tight_layout()
#plt.savefig('./isosig_2dheatmap_tet.png')

#%% #K_Means on the embedded spaces
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

#Run K-Means Clustering - manifolds
kmeans_manifold = KMeans(n_clusters=len(mflds_names)).fit(np.concatenate(embeddings_2))  
manifold_randindex = adjusted_rand_score([i for i, sublist in enumerate(embeddings_2) for _ in sublist], kmeans_manifold.labels_)

#Run K-Means Clustering - number of tetrahedra
kmeans_tet = KMeans(n_clusters=len(embeddings_tetsort)).fit(np.concatenate(embeddings_tetsort)) 
tet_randindex = adjusted_rand_score([i for i, sublist in enumerate(embeddings_tetsort) for _ in sublist], kmeans_tet.labels_)
 
print(f'Rand Index Scores:\nManifold Clustering: {manifold_randindex}\nNumber of Tetrahedra Clustering: {tet_randindex}')

#%% #Compare w2v simularities and path length
simularities, path_lens = [], []
sample_number = int(1e5)
for m_idx in np.random.choice(range(len(minimising_paths)),size=sample_number,replace=True):
    p_idx = np.random.choice(range(len(minimising_paths[m_idx])))
    s_idx = np.random.choice(range(len(minimising_paths[m_idx][p_idx])-1))
    simularities.append(w2v_100.wv.similarity(minimising_paths[m_idx][p_idx][s_idx],minimising_paths[m_idx][p_idx][-1]))
    path_lens.append(len(minimising_paths[m_idx][p_idx])-s_idx-1)
    
pmcc = np.corrcoef(simularities, path_lens)[0,1]
print(f'Embedding simularity PMCC: {pmcc}')

from scipy.stats import spearmanr
SRCC  = spearmanr(simularities,path_lens)
print(f'SRCC: {SRCC}')

'''
Results:
Some preliminary results for a word2vec embedding of the isosigs according to minimising paths are provided here:
    
2d Embedding Rand Index Scores:
Manifold Clustering: -5.4040125218670195e-05
Number of Tetrahedra Clustering: 0.007613034995845875
...i.e. the k-means clustering does not significantly correlate with a clustering according to either: 1) the manifold the isosig corresponds to, or 2) the number of tetrahedra in the isosig.

100d Rand Index Scores:
Manifold Clustering: -0.0014572471937427916
Number of Tetrahedra Clustering: 0.12437180461731005
...i.e. behaviour marginally better for the 100-dimensional embedding, but still not significant.

Correlating theword2vec embedding similarity between isosigs to their distnace along a minimising path is surprisingly good though:
100d Embedding simularity-pathlen PMCC: -0.6226476817441433 (pearsons)
100d Embedding simularity-pathlen SRCC: -0.5728556709815792 (spearmans)
(1e5 samples)
'''

