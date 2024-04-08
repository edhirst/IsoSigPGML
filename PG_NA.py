'''Script to perform network analysis of the pre-generated PG graphs'''
#Import Libraries
import sys
import regina
import numpy as np
import networkx as nx
import pickle
from math import comb

#Define manifolds
mflds = ['cMcabbgqs','cMcabbjaj','cMcabbgqw','eLAkbcbddhhjhk','cMcabbjqw','gvLQQedfedffrwawrhh','fvPQcdecedekrsnrs','jLvAzQQcfeghighiiuquanobwwr']
mflds_names = ['S3','S2xS1','RP3','L71','L72','T3','PHS','H_SC']

#Select manifolds and depths to run
mfld_idx = 0 #int(sys.argv[1])
depth_choices = [1,2] #eval(sys.argv[2])

#Initialise output file
if depth_choices[0] == 1:
    with open(f'{mflds_names[mfld_idx]}/{mflds_names[mfld_idx]}_NA.txt','w') as file:
        file.write(f'{mflds_names[mfld_idx]} Network Analysis: (depth, (#triangulations, density), degree distribution, clustering (tri,squ), min cycle basis info, wiener index, EV centrality (max,range)), NumTetrahedra (index of min, min, unique distribution)\n')

#Perform NA depth by depth
for depth_choice in depth_choices:
    #Import PG 
    PG_import = pickle.load(open(f'{mflds_names[mfld_idx]}/PG_{mflds_names[mfld_idx]}_({depth_choice},23).pickle', 'rb'))
    #Compute NA
    basic_data = [PG_import.number_of_nodes(),nx.density(PG_import),nx.degree_histogram(PG_import),sum(nx.clustering(PG_import).values())/float(len(PG_import)),sum(nx.square_clustering(PG_import).values())/float(len(PG_import))]
    cycle_basis = nx.minimum_cycle_basis(PG_import)
    cycle_lengths, freqs = np.unique(np.array(list(map(len,cycle_basis))), return_counts=True)
    del(cycle_basis)
    WIndex = nx.wiener_index(PG_import)
    WIndex_norm = WIndex/comb(len(PG_import),2)
    EVCentrality = nx.eigenvector_centrality(PG_import,max_iter=1000)
    max_EVC = max(EVCentrality, key=EVCentrality.get)
    diff_EVC = max(EVCentrality.values())-min(EVCentrality.values())
    del(EVCentrality,PG_import)
    #Import IsoSigs (truncated T_list)
    with open(f'{mflds_names[mfld_idx]}/T_list_{mflds_names[mfld_idx]}.txt','r') as file:
        T_list = []
        for line in file.readlines()[:basic_data[0]]: #...imported T_list is as long as the max graph depth generated, so for a lower depth just truncate to the first n entries of list where the graph at this depth has n nodes!
            T_list.append(line.strip('\n'))
    numtetrahedra = [regina.engine.Triangulation3.fromSig(sig).countTetrahedra() for sig in T_list]
    numtet_values, numtet_counts = np.unique(numtetrahedra,return_counts=True)
    numtet_data = ((numtetrahedra.index(min(numtetrahedra)),min(numtetrahedra)), (numtet_values.tolist(),numtet_counts.tolist()))
    del(T_list)
    
    #Save analysis
    with open(f'{mflds_names[mfld_idx]}/{mflds_names[mfld_idx]}_NA.txt','a') as file:
        file.write(f'{depth_choice}\n')
        file.write(f'({basic_data[0]}, {basic_data[1]})\n')
        file.write(f'{basic_data[2]}\n')
        file.write(f'({basic_data[3]}, {basic_data[4]})\n')
        file.write(f'{np.asarray((cycle_lengths, freqs)).T.tolist()}\n')
        file.write(f'({WIndex}, {WIndex_norm})\n') 
        file.write(f'({max_EVC}, {diff_EVC})\n')
        file.write(f'{numtet_data}\n\n')
    del(numtetrahedra)
