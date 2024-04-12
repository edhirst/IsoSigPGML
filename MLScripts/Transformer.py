'''Script for ML classifciation with transformers'''
#Import libraries
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef
from torch.utils.data import TensorDataset, DataLoader
import time
start = time.time()

'''
# Sample data -- available for testing
data1 = ['kLLLMAAkcddfgfhiijjtgfarfmarqk', 'kLLLAAPkcddegghjiijawsdgrarcvj', 'kLALvPQkcbbejigijjihhksanonpps', 'kLLMvPQkcbefjgiihijabaqafuukdw', 'kLLLAPAkcedfggihhjjlafalfxabxj', 'kLLLLQPkadfhgfihjjjnamlkfdqrrr', 'kLLALPPkbdcdfihihjjevsscagcbew', 'kLLMwQzkbeefdhghijjaaaarafxsxj', 'kLLALLQkaedefiijhjjnaaqwgwogok', 'kLLLLAQkbdfeghhjijjavdnnbbdcae']
data2 = ['kLLwLPQkcceiihghjjjajassjsnqhh', 'kLLLwPQkaeefhjhihjjntpgipiiijj', 'kLLLLMQkbegghiijjijlaaivfvqegu', 'kLLALMMkcedchfhiijjaavvhvtasfr', 'kLLvzQQkcdiigfjhhjjawabsgejvep', 'kLLALLQkccdehhgjjijakvhqhvagfo', 'kLLLMPPkbdeefgijijjerqqrgtmdmf', 'kLLLMPPkbdeefiihhjjevaavaavfrj', 'kLLMvAQkacdehfjijjijkachcachhn', 'kLLAMzAkaeddeghhijjnqaaolubqqn']
labels = [0] * len(data1) + [1] * len(data2)
data = data1 + data2
'''

#Import the same-length isosig manifold data
mflds_names = ['S3','S2xS1','RP3','L71','L72','T3','PHS','H_SC']

def importdataSL(mfld_idx):
    isosigs = []
    with open(f'../Data/Databases/ManifoldData_SL/{mflds_names[mfld_idx]}_23.txt','r') as file:
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
    
#Data set-up
choices = [3,4] #...select the manifolds to include in the classification (index 0-7)

#Set-up data
data = [sig for choice_idx in choices for sig in Manifold_isosigs[choice_idx]]
labels = [choices.index(choice_idx) for choice_idx in choices for sig in Manifold_isosigs[choice_idx]]
train_texts, test_texts, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=57)

#Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

#Tokenize the input data
train_tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in train_texts]
test_tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in test_texts]

#Pad tokenized sequences
max_len = max(len(tokenized_text) for tokenized_text in train_tokenized_texts + test_tokenized_texts)
train_padded_tokenized_texts = [tokenized_text + [0] * (max_len - len(tokenized_text)) for tokenized_text in train_tokenized_texts]
test_padded_tokenized_texts = [tokenized_text + [0] * (max_len - len(tokenized_text)) for tokenized_text in test_tokenized_texts]

#Convert tokenized sequences to torch tensors
train_input_ids = torch.tensor(train_padded_tokenized_texts)
test_input_ids = torch.tensor(test_padded_tokenized_texts)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

#Create dataloader
train_dataset = TensorDataset(train_input_ids, train_labels)
test_dataset = TensorDataset(test_input_ids, test_labels)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#Define classifier layer to convert the output
classifier = nn.Linear(768, 2)  #...use the standard BERT output size of 768

#Define loss function and optimizer
criterion = nn.CrossEntropyLoss() #...softmax not needed
optimizer = torch.optim.Adam([
    {'params': model.parameters(),      'lr': 2e-5},
    {'params': classifier.parameters(), 'lr': 1e-3}
])

#Train transformer
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    classifier.train() 
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)[1]  #...just use the final 'pooled' output for each input sequence
        outputs = classifier(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1) 
        total += labels.size(0)                       #...count number of inputs in batch
        correct += (predicted == labels).sum().item() #...count number of correctly classified inputs in batch
    
    train_accuracy = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Train Accuracy: {train_accuracy}")

#Test transformer
model.eval()
classifier.eval() 
predictions, truth = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)[1]
        logits = classifier(outputs) 
        predictions += list(torch.argmax(logits, dim=1).numpy())
        truth += list(labels.numpy())
        
accuracy = accuracy_score(truth, predictions)
mcc = matthews_corrcoef(truth, predictions)

print(f"Accuracy: {accuracy}, MCC: {mcc}")

#Output runtime
end = time.time()
print(f'Time Elapsed: {end-start} seconds')      
        
        
