'''Script to analyse the PG network data
To run, run cell 1, then either cell 2 or 3 to import analysis for snappy cusped census PGs, or focused manifold PGs; follwoing cells can be run sequentially to reproduce plots in the paper.
'''
import numpy as np
import matplotlib.pyplot as plt

filepath_root = './Data/PGData/'
snappy = False

#%% #Define snappy filepaths
depths, move_sets = [3,3], [[1,4],[2,3]]
move_strs = []
filepaths = []
for depth, moves in zip(depths, move_sets):
    move_str = str()
    for i in moves: move_str += str(i)
    move_strs.append(move_str)
    filepaths.append(filepath_root+f'snappycusp/CuspedCensusL8_NA_{depth}_{move_str}.txt')
del(depth,moves,i,move_str)

#%% #Define deep PG filepaths
mflds_names = ['S3','S2xS1','RP3','L71','L72','T3','PHS','H_SC']
mflds_strnames = [r'$S^3$',r'$S^2 \times S^1$',r'$\mathbb{RP}^3$',r'$L(7,1)$',r'$L(7,2)$',r'$T^3$',r'$PHS$',r'$H_{SC}$']
#mfld_linestyles = ['-',':','-','-','-','--','-.','-.']
snappy = False
filepaths = []
for mfld in mflds_names:
    filepaths.append(filepath_root+'DeepPG/'+mfld+'/'+mfld+'_NA.txt')
del(mfld)

#%% #Import selected datasets
nodes, density, degrees, clustering, cycles, wiener, ev_centrality, minnumtet, numtet = [], [], [], [], [], [], [], [], []

#Loop through selected flepaths importing the respective data
for fpath in filepaths:
    nodes.append([])
    density.append([])
    degrees.append([])
    clustering.append([])
    cycles.append([])
    wiener.append([])
    ev_centrality.append([]) 
    minnumtet.append([])
    numtet.append([])
    
    with open(fpath,'r') as file:
        for idx, line in enumerate(file.readlines()[1:]):
            if   idx%9 == 1:
                temp = eval(line)
                nodes[-1].append(temp[0])                               #...number of nodes
                density[-1].append(temp[1])                             #...graph density
            elif idx%9 == 2: degrees[-1].append(eval(line))             #...list of node frequencies for each list index as the degree
            elif idx%9 == 3: clustering[-1].append(list(eval(line)))    #...(triangle, square) clustering coeffiicients
            elif idx%9 == 4: cycles[-1].append(eval(line))              #...for min cycle basis (cycle lengths, frequencies of those lengths)
            elif idx%9 == 5: wiener[-1].append(list(eval(line)))        #...(wiener index, normalised wiener index)
            elif idx%9 == 6: ev_centrality[-1].append(list(eval(line))) #...(index of most central node, difference max to min centrality)
            elif idx%9 == 7: 
                temp = eval(line)
                minnumtet[-1].append(list(temp[0]))                     #...(index of minimum number of tetrahedra triangulation, minimum number of tetrahedra)
                numtet[-1].append(list(temp[1]))                        #...(number of tetrahedra unique values, frequencies)
    del(fpath,file,idx,line,temp)

#Convert to arrays (different depths NA computed to in mfld case so can't make into arrays)
max_degree = max([max(map(len,i)) for i in degrees]) #...actually the (maximum degree - 1) due to indexing
max_numtets = max([max(map(len,i[0])) for i in numtet])
if snappy:
    nodes = np.array(nodes)
    density = np.array(density)
    degrees = np.array([[i+[0]*(max_degree-len(i)) for i in j] for j in degrees]) #...postpad all distributions with zeros
    clustering = np.array(clustering)
    wiener = np.array(wiener)
    ev_centrality = np.array(ev_centrality)
    minnumtet = np.array(minnumtet)
    numtet = np.array([[[i[0]+[0]*(max_numtets-len(i[0])),i[1]+[0]*(max_numtets-len(i[1]))] for i in j] for j in numtet]) #...postpad all distributions with zeros

#################################################################################
#%% #Node Analysis
#Histograms
plt.figure('Node Histogram')
print('Node Count Analysis...')
for idx in range(len(depths)):
    values, freqs = np.unique(nodes[idx],return_counts=True)
    plt.scatter(values,freqs,alpha=0.5,label=f'{depths[idx]}_{move_strs[idx]}')
    #plt.hist(nodes[idx],bins=np.array(range(max(nodes[idx])+2))-0.5,label=f'{depths[idx]}_{move_strs[idx]}')
    print(f'Depth: {depths[idx]}, Moves: {move_sets[idx]}\n# Unique: {len(values)}\nStats (min,mean,max): {(min(values),np.mean(nodes[idx]),max(values))}\n')
    #print(f'Unique: {values}\nCounts: {freqs}\n')
plt.xlabel('Number of Nodes') 
plt.ylabel('Frequency')
#plt.ylim(0)
leg=plt.legend(loc='upper right')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.grid()
plt.tight_layout()
#plt.savefig('Snappy_nodes_histogram.pdf')
del(idx,leg,lh)

#%% #Density Analysis
plt.figure('Density Histogram')
print('Density Analysis...')
for idx in range(len(depths)):
    values, freqs = np.unique(density[idx],return_counts=True)
    plt.scatter(values,freqs,alpha=0.5,label=f'{depths[idx]}_{move_strs[idx]}')
    print(f'Depth: {depths[idx]}, Moves: {move_sets[idx]}\n# Unique: {len(values)}\nStats (min,mean,max): {(min(values),np.mean(density[idx]),max(values))}\n')
    #print(f'Unique: {values}\nCounts: {freqs}\n')
plt.xlabel('Density') 
plt.ylabel('Frequency')
#plt.ylim(0)
leg=plt.legend(loc='upper right')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.grid()
plt.tight_layout()
#plt.savefig('Snappy_density_histogram.pdf')
       
#%% #Degree Analysis
#Degree Bar Charts
data_idx = 0 #...choose which imported dataset to analyse
print('Degree Analysis...')
print(f'Depth: {depths[data_idx]}, Moves: {move_sets[data_idx]}')
print(f'Frequency stats (min,mean,max):\n{np.array([np.amin(degrees[data_idx,:,:],0),np.mean(degrees[data_idx,:,:],0),np.amax(degrees[data_idx,:,:],0)])}')

plt.figure('Degree Histogram')
#Stats --> not that useful a visualisation
#plt.bar(range(max_degree),np.amax(degrees[data_idx,:,:],0),label='max')
#plt.bar(range(max_degree),np.mean(degrees[data_idx,:,:],0),label='mean')
#plt.bar(range(max_degree),np.amin(degrees[data_idx,:,:],0),label='min')

#Total
#plt.bar(range(max_degree),np.sum(degrees[data_idx,:,:],0))
#plt.xticks(range(max_degree))
width = 0.4    #...bar width for side-by-side plotting  
plt.bar(range(max_degree),np.mean(degrees[0,:,:],0),width,label=f'{depths[0]}_{move_strs[0]}')
plt.bar(np.array(range(max_degree),dtype=int)+width,np.mean(degrees[1,:,:],0),width,label=f'{depths[1]}_{move_strs[1]}')
plt.xlabel('Degree') 
plt.xticks(np.array(range(max_degree),dtype=int) + width / 2, range(max_degree))
plt.ylabel('Mean Frequency')
#plt.ylim(0)
#plt.yscale('log') #...can't have with the ylim(0)
plt.legend(loc='upper right')
plt.grid()
plt.tight_layout()
#plt.savefig(f'Degree_histogram_{depths[data_idx]}_{move_strs[data_idx]}.pdf')
#plt.savefig('Snappy_degree_histogram.pdf')

#%% #Clustering analysis
for data_idx in range(len(depths)):
    print(f'Depth: {depths[data_idx]}, Moves: {move_sets[data_idx]}')
    #Clustering stats
    print(f'Frequency stats (min,mean,max):\nTri: {min(clustering[data_idx,:,0]),np.mean(clustering[data_idx,:,0]),max(clustering[data_idx,:,0])}, Sq: {min(clustering[data_idx,:,1]),np.mean(clustering[data_idx,:,1]),max(clustering[data_idx,:,1])}')
    #Return number of manifolds with 0 clustering
    print(f'Zero frequencies:\nTri: {sum(clustering[data_idx,:,0]==0)}, Squ: {sum(clustering[data_idx,:,1]==0)}\n')

plt.figure('Clustering Histogram')
for idx in range(len(depths)):
    values, freqs = np.unique(clustering[idx,:,1],return_counts=True)
    plt.scatter(values,freqs,alpha=0.5,label=f'{depths[idx]}_{move_strs[idx]}')
plt.xlabel('Square Clustering Coefficient') 
plt.ylabel('Frequency')
#plt.ylim(0)
leg=plt.legend(loc='upper right')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.grid()
plt.tight_layout()
#plt.savefig('Snappy_sqclust_histogram.pdf')

#%% #Shortest-path (Wiener index) analysis
for data_idx in range(len(depths)):
    print(f'Depth: {depths[data_idx]}, Moves: {move_sets[data_idx]}')
    print(f'Frequency stats (min,mean,max):\nFull: {min(wiener[data_idx,:,0]),np.mean(wiener[data_idx,:,0]),max(wiener[data_idx,:,0])}, Norm: {min(wiener[data_idx,:,1]),np.mean(wiener[data_idx,:,1]),max(wiener[data_idx,:,1])}')

plt.figure('Wiener Histogram')
for idx in range(len(depths)):
    values, freqs = np.unique(wiener[idx,:,1],return_counts=True)
    plt.scatter(values,freqs,alpha=0.5,label=f'{depths[idx]}_{move_strs[idx]}')
plt.xlabel('Normalised Wiener Index') 
plt.ylabel('Frequency')
#plt.ylim(0)
leg=plt.legend(loc='upper right')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.grid()
plt.tight_layout()
#plt.savefig('Snappy_wiener_histogram.pdf')

#%% #Eigenvector centrality analysis
for data_idx in range(len(depths)):
    print(f'Depth: {depths[data_idx]}, Moves: {move_sets[data_idx]}')
    #Clustering stats
    print(f'Frequency stats (min,mean,max):\nNode: {min(ev_centrality[data_idx,:,0]),np.mean(ev_centrality[data_idx,:,0]),max(ev_centrality[data_idx,:,0])}, Range: {min(ev_centrality[data_idx,:,1]),np.mean(ev_centrality[data_idx,:,1]),max(ev_centrality[data_idx,:,1])}')
    #Return number of manifolds with 0 clustering
    print(f'Initial most central node:\n: {sum(ev_centrality[data_idx,:,0]==0)}\n')

plt.figure('Centrality Histogram')
for idx in range(len(depths)):
    values, freqs = np.unique(ev_centrality[idx,:,1],return_counts=True)
    plt.scatter(values,freqs,alpha=0.5,label=f'{depths[idx]}_{move_strs[idx]}')
plt.xlabel('Eigenvector Centrality Range') 
plt.ylabel('Frequency')
#plt.ylim(0)
leg=plt.legend(loc='upper right')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.grid()
plt.tight_layout()
#plt.savefig('Snappy_centrality_histogram.pdf')

#%% #Cycle analysis
basis_info = []
for data_idx in range(len(depths)):
    print(f'Depth: {depths[data_idx]}, Moves: {move_sets[data_idx]}')
    
    #Total frequencies of each cycle length (only lengths 4 and 6)
    cycle_lengths = np.array([0,0],dtype=int)
    for manifold in cycles[data_idx]:
        for cycle_info in manifold:
            if cycle_info[0] == 4:
                cycle_lengths[0] += cycle_info[1]
            if cycle_info[0] == 6:
                cycle_lengths[1] += cycle_info[1]
    print(cycle_lengths/len(cycles[data_idx]))
    
    #Cycle basis sizes
    basis_sizes = [sum([cycle_size[0]*cycle_size[1] for cycle_size in manifold]) for manifold in cycles[data_idx]]
    basis_sizes_unique, basis_sizes_counts = np.unique(basis_sizes,return_counts=True)
    basis_info.append([basis_sizes_unique,basis_sizes_counts])
del(manifold,cycle_info,basis_sizes,basis_sizes_unique,basis_sizes_counts)

plt.figure('Cycle Histogram')
for idx in range(len(depths)):
    plt.scatter(basis_info[idx][0],basis_info[idx][1],alpha=0.5,label=f'{depths[idx]}_{move_strs[idx]}')
plt.xlabel('Minimum Cycle Basis Length') 
plt.ylabel('Frequency')
#plt.ylim(0)
leg=plt.legend(loc='upper right')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.grid()
plt.tight_layout()
#plt.savefig('Snappy_cycle_histogram.pdf')

#%% #Number of tetrahedra analysis
#Distributions of initial num tet
print('Initial number of tetrahedra...')
for idx in range(len(depths)):
    print(f'Depth: {depths[idx]}, Moves: {move_sets[idx]}')
    print(f'(Value,Count): {np.unique(numtet[idx,:,0,0],return_counts=True)}')
    
#Histogram of avg number tetrahedra
plt.figure('Num Tet Histogram')
for idx in range(len(depths)):
    avg=np.array([np.dot(i[0],i[1])/np.sum(i[1]) for i in numtet[idx,:]])
    avg_uniq, avg_count = np.unique(avg,return_counts=True)
    plt.scatter(avg_uniq,avg_count,alpha=0.5,label=f'{depths[idx]}_{move_strs[idx]}')
plt.xlabel('Average Number of Tetrahedra') 
plt.ylabel('Frequency')
#plt.ylim(0)
leg=plt.legend(loc='upper left')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.grid()
plt.tight_layout()
#plt.savefig('Snappy_numtet_histogram.pdf')
del(avg,avg_uniq,avg_count)

#Total num nodes vs num initial tet
plt.figure('Node v initial numtet')
#Plot with offset so no overlap
plt.scatter(numtet[0,:,0,0]-0.1,nodes[0,:],alpha=0.5,label=f'{depths[0]}_{move_strs[0]}')
plt.scatter(numtet[1,:,0,0]+0.1,nodes[1,:],alpha=0.5,label=f'{depths[1]}_{move_strs[1]}')
plt.xlabel('Number of Initial Tetrahedra') 
plt.ylabel('Number of Nodes')
#plt.ylim(0)
leg=plt.legend(loc='upper left')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
plt.grid()
plt.tight_layout()
#plt.savefig('Snappy_nodenumtet_histogram.pdf')  

   

