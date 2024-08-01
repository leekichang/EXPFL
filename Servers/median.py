import os
import sys
sys.path.insert(0, '/home/eis/disk5/Kichang/EXPFL')

from Servers import *

class MedianServer(BaseServer):
    def __init__(self, args):
        super(MedianServer, self).__init__(args)
    
    def aggregate(self, sampled_clients):
        global_state_dict = self.global_model.state_dict()

        new_state_dict = {}

        param_lists = {k: [] for k in global_state_dict.keys()}

        for client_idx in sampled_clients:
            client_state_dict = self.clients[client_idx].model.state_dict()
            for k in global_state_dict.keys():
                param_lists[k].append(client_state_dict[k].cpu().numpy())

        for k in global_state_dict.keys():
            param_array = np.array(param_lists[k])
            median_param = torch.tensor(np.median(param_array, axis=0), dtype=torch.float32)
            new_state_dict[k] = median_param

        self.global_model.load_state_dict(new_state_dict)
        
        
if __name__ == '__main__':
    import utils
    args = utils.parse_args()
    server = MedianServer(args)
    print('MEDIAN TEST')
    sampled_clients = server.sample_clients(5)
    for client_idx in sampled_clients:
        server.clients[client_idx].train()
        server.clients[client_idx].test()
    server.aggregate(sampled_clients)
    server.global_test()