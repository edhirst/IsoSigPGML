'''Script for network analysis of Cusped Pachner graphs
The second cell generates the subset of the orientable cusped census (OCC) that is analysed.
Can comment out this cell and run the whole script to run the analysis.
'''
#Import Libraries
import sys
import snappy
import regina
import numpy as np
import networkx as nx
from math import comb

#%% #Compute isosigs and number of tetrahedra for snappy cusped census, look at overlap of closed census unfilled unique with cusped census
cuspisosigs = [[] for i in range(10)]
for census_num in range(61911):
    #if census_num % 1000 == 0: print(census_num)
    mfld = snappy.Triangulation(snappy.OrientableCuspedCensus[census_num])
    isosig = mfld.triangulation_isosig(decorated=False)
    cuspisosigs[mfld.num_tetrahedra()].append(isosig)

#Print number of cusped manifolds with each number of tetrahedra (as list index)
print(list(map(len,cuspisosigs)))

#Restrict to cusped manifolds with at most 7 tetrahedra, and save to an output file
CuspedCensusL8 = cuspisosigs[:8]
print(len(CuspedCensusL8[-1]),snappy.Triangulation(CuspedCensusL8[-1][-1]).num_tetrahedra())

with open('./Data/Databases/CuspedCensusL8.txt','w') as file:
    for nt in CuspedCensusL8:
        for sig in nt:
            file.write(sig+'\n')

#%% #Set hyperparameters
depth      = int(sys.argv[1])  #3
moves_list = eval(sys.argv[2]) #[2,3]
start      = int(sys.argv[3])  #0

#Define the function to generate the Pachner Graph for a chosen manifold to a specified depth
def PachnerGraph(isosig,ml=3,moves=[2,3]):
    '''
    inputs: the isosig to triangulate, the depth (number of moves) to perform, the list of moves to consider (1 = 1-4, 2 = 2-3, 3 = 3-2, 4 = 4-1), whether to print progress
    outputs: Pachner graph (networkx object), list of triangulation isosignatures (list index = graph node number)
    '''
    moves_limit = ml #...extract the depth to perform moves to
    #Retrieve manifold, triangulate, and convert to regina triangulation (via isosig)
    #if isinstance(census_num,str): T=regina.engine.Triangulation3.fromSig(census_num)
    T=regina.engine.Triangulation3.fromSig(isosig)
    #print(list(T.vertices()),'\n\n',list(T.edges()),'\n\n',list(T.triangles()),'\n\n',list(T.tetrahedra()))
    PG = nx.empty_graph()                 #...define the Pachner graph
    PG.add_node(0,label=T.isoSig())       #...add the initial node for the starting manifold 
    T_list, T_stage = [T.isoSig()], [0,1] #...intialise the list of triangulations (as isosigs), and the list of the indices to mark the beginning&end of each stage (so consider moves on triangulations from previous stage)

    #Define function to update the graph and list
    def PG_update():
        #If the produced triangulation is new add it to the graph and list
        if newsig not in T_list: 
            T_list.append(newsig)
            PG.add_node(len(T_list)-1,label=newsig)  #...add a new node for the new triangulation
            PG.add_edge(current_node,len(T_list)-1) #...connect the new triangulation to the previous one with an edge to note what object moved on
        #If the produced triangulation is old, add an edge connecting the nodes related to this move
        else:
            new_edge = (min(current_node,T_list.index(newsig)),max(current_node,T_list.index(newsig)))
            #If the edge doenst already exists, add it
            if not PG.has_edge(*new_edge): PG.add_edge(*new_edge)
        return

    #Generate the Pachner graph (up to the specified depth)
    for move in range(moves_limit):
        for idx, sig in enumerate(T_list[T_stage[0]:T_stage[1]]):
            #Redefine the triangulation in consideration
            tri = regina.engine.Triangulation3.fromSig(sig)
            current_node = T_stage[0]+idx 
            
            if 4 in moves:
                #Loop over vertices
                for i,v in enumerate(tri.vertices()):
                    #Perform all 4-1 moves
                    if tri.pachner(v,perform=False):
                        newT = regina.engine.Triangulation3.fromSig(sig)
                        newT.pachner(newT.vertices()[i]) 
                        newsig = newT.isoSig()
                        #Update the graph
                        PG_update()

            if 3 in moves:
                #Loop over edges
                for i,e in enumerate(tri.edges()):
                    #Perform all 3-2 moves
                    if tri.pachner(e,perform=False):
                        newT = regina.engine.Triangulation3.fromSig(sig)
                        newT.pachner(newT.edges()[i]) 
                        newsig = newT.isoSig()
                        #Update the graph
                        PG_update()

            if 2 in moves:
                #Loop over triangles
                for i,t in enumerate(tri.triangles()):
                    #Perform all 2-3 moves
                    if tri.pachner(t,perform=False):
                        #If the move can be performed, do it and save the new isosig
                        newT = regina.engine.Triangulation3.fromSig(sig)
                        newT.pachner(newT.triangles()[i]) 
                        newsig = newT.isoSig()
                        #Update the graph
                        PG_update()
            
            if 1 in moves:
                #Loop over tetrahedra
                for i,h in enumerate(tri.tetrahedra()):
                    #Perform all 1-4 moves
                    if tri.pachner(h,perform=False):
                        newT = regina.engine.Triangulation3.fromSig(sig)
                        newT.pachner(newT.tetrahedra()[i]) 
                        newsig = newT.isoSig()
                        #Update the graph
                        PG_update()
                    
        #Update the T_stage list (such that the next iteration only moves on newsigs produced in this stage)
        T_stage[0] = T_stage[1]
        T_stage[1] = len(T_list)
        
    return PG, T_list

def networkanalysis(isosig):
    PG, T_list = PachnerGraph(isosig,ml=depth,moves=moves_list)
    cycle_basis = nx.minimum_cycle_basis(PG)
    cycle_lengths, freqs = np.unique(np.array(list(map(len,cycle_basis))), return_counts=True)
    WIndex = nx.wiener_index(PG)
    EVCentrality = nx.eigenvector_centrality(PG,max_iter=1000)
    max_EVC = max(EVCentrality, key=EVCentrality.get)
    numtetrahedra = [regina.engine.Triangulation3.fromSig(sig).countTetrahedra() for sig in T_list]
    numtet_values, numtet_counts = np.unique(numtetrahedra,return_counts=True)
    numtet_data = ((numtetrahedra.index(min(numtetrahedra)),min(numtetrahedra)), (numtet_values.tolist(),numtet_counts.tolist()))
    return (isosig, PG.number_of_nodes(), nx.density(PG), nx.degree_histogram(PG), sum(nx.clustering(PG).values())/float(len(PG)), sum(nx.square_clustering(PG).values())/float(len(PG)), np.asarray((cycle_lengths, freqs)).T.tolist(), WIndex, WIndex/comb(len(PG),2), max_EVC, max(EVCentrality.values())-min(EVCentrality.values()), numtet_data, numtetrahedra[0])

########################################################
if __name__ == '__main__':
    #Import libraries
    from multiprocessing import Pool
    move_str = str()
    for i in moves_list: move_str += str(i)
       
    CCL8_isosigs = []
    #Import isosig data
    with open('./Data/Databases/CuspedCensusL8.txt','r') as file:
        for line in file.readlines():
            CCL8_isosigs.append(line.strip('\n'))

    #Initialise output file:
    if start == 0:
        with open(f'./Data/PGData/snappycusp/CuspedCensusL8_NA_{depth}_{move_str}.txt','w') as file:
            file.write('Snappy OrientedCuspedCensus (with number tetrahedra < 8) Network Analysis ((initial # tetrahedra, isosig), (#triangulations, density), degree distribution, clustering (tri,squ), min cycle basis info, wiener index, EV centrality (max,range)), NumTetrahedra (index of min, min, unique distribution)\n')

    #Loop over the census (of 4815) computing NA properties
    with Pool(12) as p:
        for output in p.imap(networkanalysis,CCL8_isosigs[start:]):
            #Save data to file
            with open(f'./Data/PGData/snappycusp/CuspedCensusL8_NA_{depth}_{move_str}.txt','a') as file:
                file.write(f'{output[12]}, {output[0]}\n')
                file.write(f'({output[1]}, {output[2]})\n')
                file.write(f'{output[3]}\n')
                file.write(f'({output[4]}, {output[5]})\n')
                file.write(f'{output[6]}\n')
                file.write(f'({output[7]}, {output[8]})\n') 
                file.write(f'({output[9]}, {output[10]})\n')
                file.write(f'{output[11]}\n\n')
                