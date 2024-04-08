'''Isosig direct classification
...to run: select the investigation to perform (adjusted with the booleans in this cell), then run cells sequentially.
'''
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from itertools import combinations

#Define the alphabet for onehot encoding
alphabet = list(string.ascii_lowercase) + list(string.ascii_uppercase) + [str(i) for i in range(10)] + ['+','-']

moves = '23'
Knot = False
Surgery = False
#Same length isosigs
if not Knot and not Surgery:
    mflds_names = ['S3','S2xS1','RP3','L71','L72','T3','PHS','H_SC']
    subfolder = 'ManifoldData'
#Knot isosigs
elif Knot:
    mflds_names = ['Unknot_c','Trefoil_c','FigureEight_c','5_1_c','5_2_c','DT6a_3_c','DT8n_1_c']
    subfolder = 'KnotData'
#Surgery isosigs (Trefoil)
elif Surgery:
    mflds_names = ['Trefoil_c','Trefoil_0','Trefoil_1']
    subfolder = 'SurgeryData'

#%% #Import data
def importdataSL(mfld_idx):
    isosigs = []
    with open('./Data/Databases/'+subfolder+f'/{mflds_names[mfld_idx]}_'+moves+'.txt','r') as file:
        for line in file.readlines():
            isosigs.append(line.strip('\n'))
    return isosigs 

#Import the respective datasets
Manifold_isosigs = []
for mfld_idx in range(len(mflds_names)):
    mfld_is = importdataSL(mfld_idx)
    Manifold_isosigs.append(mfld_is)
print(f'Dataset lengths: {list(map(len,Manifold_isosigs))}')
del(mfld_idx,mfld_is)

#%% #Define ML functions
#Define the neuron activation function to use
def act_fn(x): #...leaky-ReLU activation
    return keras.activations.relu(x,alpha=0.01) 

#Define NN function (build & train)
def TrainNN(layer_sizes, train_data, test_data, bs=64, num_epochs=30, lr=1e-3, dp=0): 
    #Setup NN
    nn = keras.models.Sequential()
    nn.add(keras.layers.InputLayer(input_shape=train_data.shape[1]))
    for layer_size in layer_sizes:
        nn.add(keras.layers.Dense(layer_size, activation=act_fn, kernel_regularizer='l2'))
        if dp: nn.add(keras.layers.Dropout(dp)) #...dropout layer to reduce chance of overfitting to training data
    nn.add(keras.layers.Dense(len(choices), activation='softmax'))
    #Compile NN
    nn.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics='accuracy') 
    #Train NN
    history = nn.fit(train_data, test_data, batch_size=bs, epochs=num_epochs, shuffle=True, validation_split=0.1, verbose=0) #, use_multiprocessing=1, workers=4)
    return nn, history

#Define function for training plots
def TrainPlots(choice_idx, save=False, num_epochs=30):
    plt.figure()
    plt.plot(range(1,num_epochs+1),np.mean(np.array([run.history['accuracy'] for run in histories[choice_idx]]),axis=0),label='acc')
    plt.plot(range(1,num_epochs+1),np.mean(np.array([run.history['val_accuracy'] for run in histories[choice_idx]]),axis=0),label='val-acc')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(range(0,num_epochs+1,5))
    plt.xlim(0,num_epochs+1)
    plt.ylim(0,1)
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()
    if save: plt.savefig('./TrainingAccuraciesPlot_###.pdf')

#%% #Run ML
#Define hyperparameters
k_cv = 5  #...number of cross-validations to run
layer_sizes = [256,128,64]
num_epochs = 30 
dropout = 0.01
learning_rate = 1e-3

BinaryMatchUpIdxs = list(combinations(range(len(mflds_names)), 2))
models, histories = [], []
accuracies, mccs, cms = [], [], []
    
#Loop over the binary classifications
for choices in BinaryMatchUpIdxs:
    print(f'Classifying: {mflds_names[choices[0]]} vs {mflds_names[choices[1]]}')

    #Set-up data (onehotencoded)
    X, Y, = [], []
    for idx, choice_idx in enumerate(choices):
        for sig in Manifold_isosigs[choice_idx]:
            word = np.zeros(len(sig)*len(alphabet)).tolist()
            for letter_idx, letter in enumerate(sig):            
                word[letter_idx*len(alphabet)+alphabet.index(letter)] = 1
            X.append(word)
            label = np.zeros(len(choices))
            label[idx] = 1.
            Y.append(label)
    
    X = np.array(X)
    Y = np.array(Y)
    
    #Shuffle the data
    X, Y = shuffle(X, Y)
    
    #Perform train:test split (currently doing validation implicitly within .fit())
    if k_cv == 1:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=None, shuffle=True)
        X_train, X_test, Y_train, Y_test = [X_train], [X_test], [Y_train], [Y_test]
    elif k_cv > 1:
        X_train, X_test, Y_train, Y_test = [], [], [], []
        s = int(np.floor(len(Y)/k_cv)) #...number of datapoints in train split
        for i in range(k_cv):
            X_train.append(np.concatenate((X[:i*s],X[(i+1)*s:])))
            Y_train.append(np.concatenate((Y[:i*s],Y[(i+1)*s:])))
            X_test.append(X[i*s:(i+1)*s])
            Y_test.append(Y[i*s:(i+1)*s])    
        del(i)
    del(idx,choice_idx,sig,word,letter_idx,letter,label)
    
    #Run NN investigation
    models.append([])
    histories.append([])
    accuracies.append([])
    mccs.append([]) 
    cms.append([])
    for run in range(k_cv):
        #Training
        m, h = TrainNN(layer_sizes,X_train[run],Y_train[run],num_epochs=num_epochs,lr=learning_rate,dp=dropout) 
        models[-1].append(m)
        histories[-1].append(h)
        print(f'Run {run+1} final validation accuracy: {h.history["val_accuracy"][-1]}')
        #Testing
        p = np.argmax(m.predict(X_test[run]),axis=1)
        true_p = np.argmax(Y_test[run],axis=1)
        accuracies[-1].append(accuracy_score(true_p,p))
        mccs[-1].append(matthews_corrcoef(true_p,p))
        cms[-1].append(confusion_matrix(true_p,p,normalize='all'))
        del(m,h,p,true_p)

#Compute average test measures
meanacc_mat = np.eye(len(mflds_names))
meanmcc_mat = np.eye(len(mflds_names))
for bd_idx in range(len(BinaryMatchUpIdxs)): 
    meanacc_mat[BinaryMatchUpIdxs[bd_idx]] = np.mean(accuracies[bd_idx])
    meanacc_mat[tuple(reversed(BinaryMatchUpIdxs[bd_idx]))] = meanacc_mat[BinaryMatchUpIdxs[bd_idx]]
    meanmcc_mat[BinaryMatchUpIdxs[bd_idx]] = np.mean(mccs[bd_idx])
    meanmcc_mat[tuple(reversed(BinaryMatchUpIdxs[bd_idx]))] = meanmcc_mat[BinaryMatchUpIdxs[bd_idx]]

#Output example training plot
ex_choice = np.random.choice(range(len(BinaryMatchUpIdxs))) 
print(f'\nExample Measures:\t{mflds_names[BinaryMatchUpIdxs[ex_choice][0]]} vs {mflds_names[BinaryMatchUpIdxs[ex_choice][1]]}\nAccuracy:\t\t\t{np.mean(accuracies[ex_choice])} $\pm$ {np.std(accuracies[ex_choice])/k_cv}\nMCC:\t\t\t\t\t{np.mean(mccs[ex_choice])} $\pm$ {np.std(mccs[ex_choice])/k_cv}\nCM: {np.mean(cms[ex_choice],0).tolist()} $\pm$ {(np.std(cms[ex_choice],0)/k_cv).tolist()}')
TrainPlots(ex_choice,num_epochs=num_epochs)


 