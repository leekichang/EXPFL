import os
import sys
sys.path.insert(0, '/home/eis/disk5/Kichang/EXPFL')

from Servers import *

class FedAvgServer(BaseServer):
    def __init__(self, args):
        super(FedAvgServer, self).__init__(args)
    
    def aggregate(self, sampled_clients):
        global_state_dict = self.global_model.state_dict()
        new_state_dict = {k: torch.zeros_like(v) for k, v in global_state_dict.items()}

        for client_idx in sampled_clients:
            client_state_dict = self.clients[client_idx].model.state_dict()
            for k in global_state_dict.keys():
                new_state_dict[k] += client_state_dict[k] / len(sampled_clients)
        self.global_model.load_state_dict(new_state_dict)
        
        
if __name__ == '__main__':
    import utils
    args = utils.parse_args()
    server = FedAvgServer(args)
    print('FEDAVG TEST')
    sampled_cliets = server.sample_clients(5)
    for client_idx in sampled_cliets:
        server.clients[client_idx].train()
        server.clients[client_idx].test()
    server.aggregate(sampled_cliets)
    server.global_test()