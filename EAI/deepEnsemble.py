import torch
import numpy as np
'''
Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
https://arxiv.org/abs/1612.01474
'''
class DeepEnsemble(object):
    def __init__(self):
        super(DeepEnsemble, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.outputs = []
        self.softmax = torch.nn.Softmax(dim=1)
        self.targets = None
        self.preds   = []
        
    def inference(self, model, dataloader, model_paths):
        self.targets = dataloader.dataset.targets.numpy()
        with torch.no_grad():
            for midx, m in enumerate(model_paths):
                model.load_state_dict(torch.load(m))
                model.to(self.device)
                model.eval()
                model_output = []
                for idx, (data, target) in enumerate(dataloader):
                    data = data.to(self.device)
                    output = model(data)
                    prob   = torch.nn.Softmax(dim=1)(output)
                    model_output.append(prob.detach().cpu().numpy())
                self.outputs.append(np.concatenate(model_output, axis=0))
            self.outputs = np.array(self.outputs)
            
    def calc_mean(self):
        self.mean_probs = np.mean(self.outputs, axis=0)
        self.preds      = np.argmax(self.mean_probs, axis=1)
        self.std_probs  = np.std(self.outputs, axis=0)
        print(self.std_probs.shape, self.preds.shape)
    
    def calc_entropy(self):
        self.entropy = -np.sum(self.mean_probs * np.log(self.mean_probs + 1e-6), axis=1)
        print('self.entropy.shape', self.entropy.shape)
    
    def __call__(self, model, dataloader, model_paths):
        self.inference(model, dataloader, model_paths)
        self.calc_mean()
        self.calc_entropy()

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/eis/disk5/Kichang/EXPFL')
    import DataManager.datamanager as dm
    from models.CNN import CNN
    model = CNN(n_class=10)
    _, testset = dm.MNIST()
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, drop_last=False)
    N_MODELS = 10
    DE = DeepEnsemble()
    model_paths = [f'./checkpoints/centralized/centralized_{i}.pt' for i in range(N_MODELS)]
    DE(model, testloader, model_paths)