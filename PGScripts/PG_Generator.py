'''Script to generate PG graphs, saving depth by depth, until timeout'''
#Import Libraries
import sys
import regina
import networkx as nx
import pickle

#Define manifolds
mflds = ['cMcabbgqs','cMcabbjaj','cMcabbgqw','eLAkbcbddhhjhk','cMcabbjqw','gvLQQedfedffrwawrhh','fvPQcdecedekrsnrs','jLvAzQQcfeghighiiuquanobwwr']
mflds_names = ['S3','S2xS1','RP3','L71','L72','T3','PHS','H_SC']

#Check initial isosigs are 1-vertex
print('Manifold | # Vertices | # Tetrahedra')
for m in mflds: 
    T=regina.engine.Triangulation3.fromSig(m)
    print(f'{mflds_names[mflds.index(m)]}\t\t{T.countVertices()}\t\t{T.countTetrahedra()}')

#Set hyperparameters
max_depth = 50
moves_list = [2,3]
IS_init = mflds[int(sys.argv[1])] #...can hardcode an index from the list here otherwise

#Define initial triangulation
T=regina.engine.Triangulation3.fromSig(IS_init)

#Generate PG
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
for depth in range(max_depth):
    for idx, sig in enumerate(T_list[T_stage[0]:T_stage[1]]):
        #Redefine the triangulation in consideration
        tri = regina.engine.Triangulation3.fromSig(sig)
        current_node = T_stage[0]+idx 

        if 4 in moves_list:
            #Loop over vertices
            for i,v in enumerate(tri.vertices()):
                #Perform all 4-1 moves
                if tri.pachner(v,perform=False):
                    newT = regina.engine.Triangulation3.fromSig(sig)
                    newT.pachner(newT.vertices()[i]) 
                    newsig = newT.isoSig()
                    #Update the graph
                    PG_update()

        if 3 in moves_list:
            #Loop over edges
            for i,e in enumerate(tri.edges()):
                #Perform all 3-2 moves
                if tri.pachner(e,perform=False):
                    newT = regina.engine.Triangulation3.fromSig(sig)
                    newT.pachner(newT.edges()[i]) 
                    newsig = newT.isoSig()
                    #Update the graph
                    PG_update()


        if 2 in moves_list:
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

        if 1 in moves_list:
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
    
    #Define filepath
    move_str = str()
    for i in moves_list: move_str += str(i)
    filepath = f'./{mflds_names[mflds.index(IS_init)]}/PG_{mflds_names[mflds.index(IS_init)]}_({depth+1},{move_str})'

    #Save graph at each new depth
    pickle.dump(PG, open(filepath+'.pickle', 'wb'))
    with open(f'./{mflds_names[mflds.index(IS_init)]}/T_list_{mflds_names[mflds.index(IS_init)]}.txt','w') as file:
        for tri in T_list:
            file.write(tri+'\n')

