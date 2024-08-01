import os
import sys
sys.path.insert(0, '/home/eis/disk5/Kichang/EXPFL')

from Servers import *

class TrimmedMeanServer(BaseServer):
    def __init__(self, args):
        super(TrimmedMeanServer, self).__init__(args)
    
    def aggregate(self, sampled_clients, trim_fraction=0.1):
        n_clients = self.n_clients
        global_state_dict = self.global_model.state_dict()
        new_state_dict = {}
        param_lists = {k: [] for k in global_state_dict.keys()}

        for cidx in sampled_clients:
            client_state_dict = self.clients[cidx].model.state_dict()
            for k in global_state_dict.keys():
                param_lists[k].append(client_state_dict[k].cpu().numpy())

        for k in global_state_dict.keys():
            param_array = np.array(param_lists[k])
            n_trim = int(trim_fraction * param_array.shape[0])
            sorted_array = np.sort(param_array, axis=0)
            trimmed_array = sorted_array if n_trim == 0 else sorted_array[n_trim: -n_trim]
            trimmed_mean_param = torch.tensor(np.mean(trimmed_array, axis=0), dtype=torch.float32)
            new_state_dict[k] = trimmed_mean_param
        self.global_model.load_state_dict(new_state_dict)
        
if __name__ == '__main__':
    import utils
    args = utils.parse_args()
    server = TrimmedMeanServer(args)
    print('TRIMMED MEAN TEST')
    sampled_clients = server.sample_clients(5)
    for client_idx in sampled_clients:
        server.clients[client_idx].train()
        server.clients[client_idx].test()
    server.aggregate(sampled_clients)
    server.global_test()