import os
import sys
sys.path.insert(0, '/home/eis/disk5/Kichang/EXPFL')
import copy
import torch
from tqdm import tqdm

import utils
from models import *
from Clients import *
from DataManager import *

class BaseServer(object):
    def __init__(self, args):
        self.args = args
        self.clients = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.setup()
        
    def setup(self):
        self.exp_name     = self.args.exp_name
        print(f"Experiment: {self.exp_name}")
        self.n_clients    = self.args.n_clients
        print(f"Number of Clients: {self.n_clients}")
        self.save_path = f'./checkpoints/{self.exp_name}'
        self.prepare_dataset()
        self.init_clients()
        self.global_model = utils.build_model(self.args)
        self.dispatch()
    
    def prepare_dataset(self):
        self.trainset, self.testset = getattr(datamanager, self.args.dataset)()
        self.client_trainsets, self.client_testsets = Dirichlet(self.trainset, 
                                                                self.testset,
                                                                self.n_clients,
                                                                self.args.alpha).split_dataset()
        
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=False, drop_last=False)
        
    def init_clients(self):
        print(f"Initializing {self.n_clients} clients")
        for cidx in tqdm(range(self.n_clients)):
            self.clients.append(self.create_client(cidx))
            self.clients[cidx].trainset = copy.deepcopy(self.client_trainsets[cidx])
            self.clients[cidx].testset  = copy.deepcopy(self.client_testsets[cidx])
            self.clients[cidx].setup()
            
    def create_client(self, client_id):
        return getattr(Clients, self.args.method)(self.args, client_id)
    
    def sample_clients(self, n_participants):
        sampled_clients_idx = torch.randperm(self.n_clients)[:n_participants]
        return sampled_clients_idx
    
    def dispatch(self):
        for cidx in range(self.n_clients):
            self.clients[cidx].model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))
            
    def aggregate(self):
        raise NotImplementedError        
    
    def global_test(self):
        correct = 0
        self.global_model = self.global_model.to(self.device)
        self.global_model.eval()
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.global_model(data)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == target).sum().item()
        print(f'Global Accuracy: {100*correct/len(self.testset)}%')
        
    def client_test(self, client_id):
        raise NotImplementedError
    
    def save_global_model(self, round):
        torch.save(self.global_model.state_dict(), f'{self.save_path }/global_{round}.pth')
        
if __name__ == '__main__':
    
    args = utils.parse_args()
    server = BaseServer(args)
    
    for client in server.clients:
        client.train()
        client.test()
    server.aggregate()
    server.global_test()