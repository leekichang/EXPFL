import torch

import utils
import Clients

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    args = utils.parse_args()
    torch.random.manual_seed(args.seed)
    client = Clients.NaiveClient(args, f'{args.exp_name}')
    for epoch in range(10):
        client.train()
        client.test()
    client.save_model(f'{args.seed}')