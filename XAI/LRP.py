import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import cv2
from PIL import Image


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/eis/disk5/Kichang/EXPFL')
    import DataManager.datamanager as dm
    from models.CNN import CNN
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    # from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    # from torchvision.models import resnet50

    # model = resnet50(pretrained=True)
    # target_layers = [model.layer4[-1]]
    # input_tensor = torch.FloatTensor(np.array(Image.open('./0_image.png'))/255).permute(2,0,1).unsqueeze(0)
    # print(input_tensor.shape)
    
    # cam = GradCAM(model=model, target_layers=target_layers)

    # targets = [ClassifierOutputTarget(281)]

    # grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    # grayscale_cam = grayscale_cam[0, :]
    # rgb_img = np.array(Image.open('./0_image.png'))/255
    # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    # print(visualization.shape)
    # Image.fromarray((visualization*255).astype(np.uint8)).save('test.png')
    # model_outputs = cam.outputs
    _, testset = dm.MNIST()
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, drop_last=False)
    for idx, (data, target) in enumerate(testloader):
        if idx == 0:
            input_tensor = data
            break
    model = CNN(n_class=10)
    model.load_state_dict(torch.load('./checkpoints/EXPFL/Naive_test.pt'))
    
    # cam = GradCAM(model=model, target_layers=[model.conv1])
    # grayscale_cam = cam(input_tensor=input_tensor)
    # grayscale_cam = grayscale_cam[0, :]
    # rgb_img = np.array(data[0].permute(1,2,0))
    
    
    
    # Plot figures
    import matplotlib.pyplot as plt
    plt.imshow(grayscale_cam)
    plt.savefig('cam.png')
    # visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    Image.fromarray((visualization*255).astype(np.uint8)).save('test.png')
    print('Done')