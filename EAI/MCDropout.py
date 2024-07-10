import torch
import numpy as np

class MCDropout(object):
    def __init__(self):
        super(MCDropout, self).__init__()
        self.device  = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.outputs = []
        self.softmax = torch.nn.Softmax(dim=1)
        self.targets = None
        self.preds   = []
    
    def inference(self, model, dataloader, iter=10):
        model.to(self.device)
        model.train() # For MC Dropout we need to set the model to evaluation mode
        self.targets = dataloader.dataset.targets.numpy()
        with torch.no_grad():
            for idx, (data, target) in enumerate(dataloader):
                data = data.to(self.device)
                batch_output = []
                for _ in range(iter):
                    output = model(data)
                    prob   = self.softmax(output)
                    batch_output.append(prob.detach().cpu().numpy())
                self.outputs.append(np.array(batch_output))
            self.outputs = np.concatenate(self.outputs, axis=1)
    
    def calc_mean(self):
        self.mean_probs = np.mean(self.outputs, axis=0)
        self.preds      = np.argmax(self.mean_probs, axis=1)
        self.std_probs  = np.std(self.outputs, axis=0)
        print(self.std_probs.shape, self.preds.shape)
    
    def calc_entropy(self):
        eps = 1e-10  # Adjust as needed based on your data
        clipped_probs = np.clip(self.mean_probs, eps, 1 - eps)

        # Calculate entropy using the correct formula
        self.entropy = -np.sum(clipped_probs * np.log(clipped_probs), axis=1)
        print('self.entropy.shape', self.entropy.shape)
    
    def __call__(self, model, dataloader, iter=10):
        self.inference(model, dataloader, iter=iter)
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
    MonteCarloDropout = MCDropout()
    MonteCarloDropout(model, testloader, iter=1)