import os
import sys
sys.path.insert(0, '/home/eis/disk5/Kichang/EXPFL')
import torch
import numpy as np
import torch.nn as nn

import utils
import Clients
from DataManager import datamanager

class DPSGDClient(Clients.NaiveClient):
    def __init__(self, args, name):
        super(DPSGDClient, self).__init__(args, name)
        self.epsilon = args.epsilon  # Epsilon for noise scale
        self.delta = args.delta  # Delta for differential privacy
        self.clip_norm = args.clip_norm  # Norm for gradient clipping
    
    def sanitize(self, tensor, eps_delta, sigma=None, clip_norm=None, add_noise=True):
        eps, delta = eps_delta
        if sigma is None:
            sigma = np.sqrt(2 * np.log(1.25 / delta)) / eps
        
        if clip_norm is None:
            clip_norm = self.clip_norm
        
        tensor_norm = torch.norm(tensor, p=2)
        if tensor_norm > clip_norm:
            tensor = tensor * (clip_norm / tensor_norm)
        
        if add_noise:
            noise = torch.normal(mean=0, std=sigma * clip_norm, size=tensor.shape).to(tensor.device)
            tensor = tensor + noise
        
        return tensor
    
    def sanitize_gradients(self):
        for p in self.model.parameters():
            if p.grad is not None:
                sanitized_grad = self.sanitize(p.grad.data, (self.epsilon, self.delta), clip_norm=self.clip_norm, add_noise=True)
                p.grad.data = sanitized_grad
                
    def train(self):
        self.model = self.model.to(self.device)
        self.model.train()
        for idx, (data, target) in enumerate(self.trainloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            loss.backward()
            
            # Sanitize gradients
            self.sanitize_gradients()
            
            self.optimizer.step()
        self.model = self.model.to('cpu')

if __name__ == '__main__':
    import utils
    import Clients
    from tqdm import tqdm
    args = utils.parse_args()
    client = Clients.BaseClient(args, 'DPSGD')
    client = DPSGDClient(args, 'DPSGD')
    client.trainset, client.testset = datamanager.MNIST()
    client.setup()
    
    for epoch in tqdm(range(10)):
        client.train()
        client.test()
    client.save_model('dpsgd_test')