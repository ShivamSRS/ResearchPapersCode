
#!wget https://raw.githubusercontent.com/tensorflow/privacy/master/privacy/analysis/rdp_accountant.py
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from IPython.display import clear_output
from numpy import linalg as LA
from rdp_accountant import compute_rdp  # pylint: disable=g-import-not-at-top
from rdp_accountant import get_privacy_spent
import warnings
warnings.filterwarnings('ignore')

"""# client class

The client class implements the update method.
I'm returning the deltas in a dict with the same keys as the state_dict of the "parent" model.

* wt1: Dict with the same keys as state_dict with the deltas of all the weights
* S: Dict with the same keys as state_dict with the L2 norms of all deltas
"""

class client():
  def __init__(self, number, loader, state_dict, batch_size = 64, epochs=2, lr=0.01):
    self.number = number
    self.model = t_model()
    self.model.load_state_dict(state_dict)
    self.criterion = nn.NLLLoss()
    self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
    self.epochs = epochs
    self.device =  device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.dataLoader = loader                                       
                                           
                                  
  #
  def update(self, state_dict):
    wght0 = state_dict
    self.model.load_state_dict(state_dict)
    self.model.to(self.device)
    running_loss = 0
    accuracy = 0
    for e in range(self.epochs):
        self.model.train()
        accuracy=0
        running_loss = 0
        for images, labels in self.dataLoader:            
            images, labels = images.to(self.device), labels.to(self.device)                       
            self.optimizer.zero_grad()            
            output = self.model.forward(images)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()            
            running_loss += loss.item()        
    S ={} 
    wt1 = {}
    for key, value in wght0.items():
      wght1[key] = self.model.state_dict()[key]  - value   
      S[key] = LA.norm(wt1[key].cpu(), 2)
    return wght1, S


class server():
  def __init__(self, number_clients, p_budget, epsilon, sigmat = 1.12):
    self.model = t_model()
    self.sigmat = sigmat   
    self.n_clients = number_clients
    self.samples = get_samples(self.n_clients)
    self.clients = list()
    for i in range(number_clients):
      loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=32, sampler=self.samples[i])
      self.clients.append(client(i, loader, self.model.state_dict()))
    self.p_budget = p_budgetp_budget# The delta bugdet that we have for the training rounds
    self.epsilon = epsilon
    self.testLoader = torch.utils.data.DataLoader(mnist_testset, batch_size=32)
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
            list(range(5, 64)) + [128, 256, 512])
    
    
  #Evaluates the accuracy of the current model with the test data.  
  def eval_acc(self):
    self.model.to(self.device)
    #print('Aqui voy!')
    running_loss = 0
    accuracy = 0
    self.model.eval()
    suma=0
    total = 0
    running_loss = 0
    for images, labels in self.testLoader:            
        images, labels = images.to(self.device), labels.to(self.device) 
        output = self.model.forward(images)             
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        total += equals.size(0)
        suma = suma + equals.sum().item()
    else:      
        print('Accuracy: ',suma/float(total))
   
  
  def sanitaze(self,mt, deltas, norms, sigma, state_dict):    
    new_dict = {}
    for key, value in state_dict.items():
      S=[]
      for i in range(len(norms)):        
        S.append(norms[i][key])
      S_value = np.median(S)      
      wt = value
      prom = 1/float(mt)       
      suma = 0
      for i in range(len(deltas)):    
        clip = (max(1, float(norms[i][key]/S_value)))            
        suma = suma + ((deltas[i][key] / clip ))
      noise = np.random.normal(0, float(S_value * sigma), size = suma.shape)      
      suma = suma.cpu().numpy()
      suma = suma*prom
      noise = noise*prom
      suma = suma + noise  
      
      suma = torch.from_numpy(suma)
      suma = wt + suma.float()
      new_dict[key] = suma
    return new_dict
    
 
  def server_exec(self,mt):    
    i=1
    while(True):
      clear_output()
      print('Comunication round: ', i)
      self.eval_acc()         
      rdp = compute_rdp(float(mt/len(self.clients)), self.sigmat, i, self.orders)
      _,delta_spent, opt_order = get_privacy_spent(self.orders, rdp, target_eps=self.epsilon)
      print('Delta spent: ', delta_spent)
      print('Delta budget: ', self.p_budget)    
      if self.p_budget < delta_spent:
        break
      Zt = np.random.choice(self.clients, mt)      
      deltas = []
      norms = []
      for client in Zt:
        #print(client.number)
        deltaW, normW = client.update(self.model.state_dict())        
        deltas.append(deltaW)
        norms.append(normW)
      #print('all updates')      
      self.model.to('cpu')
      new_state_dict = self.sanitaze(mt, deltas, norms, self.sigmat, self.model.state_dict())
      #print('sanitaze')
      self.model.load_state_dict(new_state_dict)
      i+=1

"""# The model class

This is the class where we define the model of our setup
"""

class t_model(nn.Module):
    def __init__(self):
        super(t_model, self).__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))        
        x = F.relu(self.fc3(x))
        return F.log_softmax(x)

#Return the samples that each client is going to have as a private training data set. This is a not overlapping set
  def get_samples(num_clients):
    tam = len(mnist_trainset)
    split= int(tam/num_clients)
    split_ini = split
    indices = list(range(tam))
    init=0
    samples = []
    for i in range(num_clients):     
      t_idx = indices[init:split]
      t_sampler = SubsetRandomSampler(t_idx)
      samples.append(t_sampler)
      init = split
      split = split+split_ini
    return samples

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, ), (0.5,))])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
device =  torch.device("cuda:0""cuda:0" if torch.cuda.is_available() else "cpu")
num_clients = 100
train_len = len(mnist_trainset)
test_len = len(mnist_testset)

#We're creating the Server class. A priv_budget of 0.001 (the max delta) and a Epsilon of 8
serv = server(num_clients, 0.001, 8)
serv.server_exec(30)

"""# Disclaimer:
I'm using the same setup as one of the experiments of the original paper. The only difference is my sigma, I have a 1.12 sigma and the authors used a 24 sigma. 

I don't know why but with a 24 sigma my setup doesn't converge. If anyone has a clue on what I'm doing wrong or what is the difference!
"""

#@title
def acc_0(model, loader, device):
    model.to(device)    
    running_loss = 0
    accuracy = 0
    model.eval()
    suma=0
    total = 0
    for images, labels in loader:            
      images, labels = images.to(device), labels.to(device)                       
      output = model.forward(images)
      ps = torch.exp(output)
      top_p, top_class = ps.topk(1, dim=1)
      equals = top_class == labels.view(*top_class.shape) 
      total += equals.size(0)
      suma = suma + equals.sum().item()
    else:
      print(suma/total)

#@title
model = t_model()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 5
trainLoader = torch.utils.data.DataLoader(mnist_trainset, batch_size=32)
testLoader = torch.utils.data.DataLoader(mnist_testset, batch_size=32)
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
acc_0(model, testLoader, device)