import torch
from EAI.DeepEnsemble import *

'''
Naive Softmax Confidence is identical to DeepEnsemble with single model.
'''

class NaiveConfidence(DeepEnsemble): 
    def __init__(self):
        super(NaiveConfidence, self).__init__()
        
    def inference(self, model, dataloader, model_paths):
        self.targets = dataloader.dataset.targets.numpy()
        with torch.no_grad():
            model.load_state_dict(torch.load(model_paths[0]))
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