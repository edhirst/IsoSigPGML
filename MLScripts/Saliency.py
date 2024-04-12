'''NN Gradient Saliency: focus on Lens space differentiation'''
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import string
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from tensorflow import keras, GradientTape, Variable

#Define the alphabet for onehot encoding
alphabet = list(string.ascii_lowercase) + list(string.ascii_uppercase) + [str(i) for i in range(10)] + ['+','-']

#Same length isosigs
mflds_names = ['L71','L72']
moves = '23'

#%% #Import data
def importdataSL(mfld_idx):
    isosigs = []
    with open(f'../Data/Databases/ManifoldData_SL/{mflds_names[mfld_idx]}_'+moves+'.txt','r') as file:
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
    nn.add(keras.layers.Dense(2, activation='softmax'))
    #Compile NN
    nn.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics='accuracy') 
    #Train NN
    history = nn.fit(train_data, test_data, batch_size=bs, epochs=num_epochs, shuffle=True, validation_split=0.1, verbose=0) #, use_multiprocessing=1, workers=4)
    return nn, history

#%% #Run ML
#Define hyperparameters
k_cv = 100  #...number of cross-validations to run
layer_sizes = [256,128,64]
num_epochs = 30 
dropout = 0.01
learning_rate = 1e-3

BinaryMatchUpIdxs = [(0,1)]
models, histories = [], []
accuracies, mccs, cms = [], [], []
average_gradients = []
    
#Set-up data (onehotencoded)
choices = [0,1]
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
X, Y = shuffle(X, Y)

#Run NN investigation
for run in range(k_cv):
    #Parition the data into (train,test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, shuffle=True)
    #Training
    m, h = TrainNN(layer_sizes,X_train,Y_train,num_epochs=num_epochs,lr=learning_rate,dp=dropout) 
    models.append(m)
    histories.append(h)
    print(f'Run {run+1} final validation accuracy: {h.history["val_accuracy"][-1]}')
    #Testing
    p = np.argmax(m.predict(X_test),axis=1)
    true_p = np.argmax(Y_test,axis=1)
    accuracies.append(accuracy_score(true_p,p))
    mccs.append(matthews_corrcoef(true_p,p))
    cms.append(confusion_matrix(true_p,p,normalize='all'))

    #Gradient Saliency
    image = Variable(X_test,dtype='float')
    with GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(image)
        predictions = m(image)
        loss = predictions
    
    #Compute the gradient of the loss wrt the input image (and convert to numpy)
    gradient = tape.gradient(loss, image)
    gradient = gradient.numpy()
    avg_grad = np.absolute(np.mean(gradient,axis=0)) 
    avg_grad = avg_grad/np.sum(avg_grad)
    
    #Save the final gradients
    #print('Average Gradients:\n',avg_grad,'\n')
    average_gradients.append(avg_grad)
    del(m,h,p,true_p)

#Average over the runs and plot
average_gradients = np.array(average_gradients)
average_gradients = np.mean(average_gradients,axis=0)
print('Final averaged gradients:\n',average_gradients)

#%% #Plot the saliency results
#Image
plt.axis('off')
plt.imshow(average_gradients.reshape((30,64)))
plt.tight_layout()
#plt.savefig('./SaliencyImage_L71L72.pdf', bbox_inches='tight')

#%% #Gradients histogram (for partitioning)
plt.figure()
bins = np.logspace(-10,0,11)
plt.hist(average_gradients,bins=bins)
plt.xlabel('Saliency Value')
plt.ylabel('Frequency')
plt.xscale('log')
plt.grid()
plt.tight_layout()
#plt.savefig('Saliency_GradientsHistogram.pdf')

#%% #Letter frequencies
salient_indices = np.where(average_gradients > 1e-4)[0]
salient_letters, salient_letter_counts = np.unique(salient_indices%len(alphabet),return_counts=True)
plt.figure()
plt.plot([alphabet[i] for i in salient_letters],salient_letter_counts)
plt.xticks([alphabet[i] for i in salient_letters])
plt.yticks(range(0,24,2))
plt.xlabel('Letter')
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
#plt.savefig('Saliency_LetterHistogram.pdf')

#%% #Isosig index frequencies
salient_indices = np.where(average_gradients > 1e-4)[0]
salient_isosigidxs, salient_isosigidx_counts = np.unique(np.floor(salient_indices/len(alphabet)),return_counts=True)
plt.figure()
plt.plot(salient_isosigidxs,salient_isosigidx_counts)
plt.yticks(range(0,30,5))
#plt.yticks(range(0,24,2))
plt.xlabel('Isosig Index')
plt.ylabel('Frequency')
plt.grid()
plt.tight_layout()
#plt.savefig('Saliency_IsosigIndexHistogram.pdf')
