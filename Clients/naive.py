import os
import sys
sys.path.insert(0, '/home/eis/disk5/Kichang/EXPFL')
import torch
import torch.nn as nn


import utils
import Clients
from DataManager import datamanager

class NaiveClient(Clients.BaseClient):
    def __init__(self, args, name):
        super(NaiveClient, self).__init__(args, name)

    def setup(self):
        self.model       = utils.build_model(self.args)
        self.criterion   = nn.CrossEntropyLoss()
        self.optimizer   = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        # self.trainset, self.testset = datamanager.MNIST()
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True , drop_last=True )
        self.testloader  = torch.utils.data.DataLoader(self.testset , batch_size=self.args.batch_size, shuffle=False, drop_last=False)
    
    def train(self):
        self.model = self.model.to(self.device)
        self.model.train()
        for idx, (data, target) in enumerate(self.trainloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            self.optimizer.step()
        self.model = self.model.to('cpu')
        
    def test(self):
        self.model = self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            print(f'CLIENT {self.name:<3} Accuracy: {100*correct/total}%')
        self.model = self.model.to('cpu')

if __name__ == '__main__':
    import utils
    import Clients
    args = utils.parse_args()
    client = Clients.BaseClient(args, 'Naive')
    client = NaiveClient(args, 'Naive')
    for epoch in range(10):
        client.train()
        client.test()
    client.save_model('test')