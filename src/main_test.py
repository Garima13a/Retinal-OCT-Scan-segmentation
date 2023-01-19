import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import MSELoss, Sequential, CrossEntropyLoss, DataParallel
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import torchvision.models as models
from torchvision.models.segmentation import deeplabv3_resnet50
from dataset import get_loader
from args import parser
from utils import get_device
from plot import plot_grid



def plot_grid(imgs, labs, masks, rows=4, cols=4, lab_normalized=False):
    fig = plt.figure(figsize=(16,16), frameon=False)
    bz = imgs.shape[0]
    bz = min([bz, rows*cols])
    if lab_normalized:
        labs = labs*imgs.shape[-1]

    for i in range(bz):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.axis('off')
        ax.imshow(imgs[i,0], cmap='gray')
        ax.matshow(masks[i], alpha=0.5, cmap=plt.get_cmap("tab10"))
        for j in range(labs.shape[2]):
            ax.plot(labs[i,:,j], alpha=0.4)
    fig.tight_layout()

    return fig

def train_epoch(net, trainloader,testloader, optimizer, criterion, device, writer, epoch_no, args):

    j = epoch_no*len(trainloader)*args.batch_size

    output_pts1 = None
    lab_reshaped = None
    labels = None
    model.train()
    for images,labels,masks in trainloader:
        j += images.shape[0]
        images1 = images.type(torch.FloatTensor)
        images1, masks1 = images.to(device), masks.to(device)
      
        
        output_pts1 = net(images1)["out"]
        # print(masks.max(), masks.min())
        # print(images.shape, masks.shape, output_pts.shape)

        loss = criterion(output_pts1, masks1)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        
        writer.add_scalar('Train Loss', loss.item(), j)


    i = epoch_no*len(testloader)*args.batch_size

    output_pts = None
    lab_reshaped = None
    labels = None
    model.eval()
    with torch.no_grad():
        for images,labels, masks in testloader:
            i += images.shape[0]
            images = images.type(torch.FloatTensor)
            images, masks = images.to(device), masks.to(device)
        
            output_pts = model(images)["out"]
            val_loss = criterion(output_pts, masks)

            # optimizer.zero_grad()

            # loss.backward()

            # optimizer.step()
            writer.add_scalar(' Test Loss', val_loss.item(), i)

    print('[Epoch %d] Train loss: %.3f Test loss: %.3f '%
                        (epoch_no + 1, loss.item(), val_loss.item()))
            
        
        
    # print(output_pts.shape)
    out_lab = output_pts.cpu().data.numpy().argmax(1)
    # print(out_lab.shape)
    # lab_reshaped = labels.cpu().data.numpy()
    # out_lab = np.concatenate((out_reshaped, lab_reshaped), 2)
        
    fig = plot_grid(images.cpu().data.numpy(), labels,  out_lab)
    writer.add_figure('Test predictions', fig, i)
    writer.flush()
    plt.close()


args    = parser.parse_args()
writer  = SummaryWriter("runs/"+ args.name)
device  = get_device()

model  = getattr(models.segmentation, args.feat_ext)(num_classes=10)
model.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = DataParallel(model)

model.to(device)
print(model)
criterion                      = CrossEntropyLoss(weight=torch.Tensor([1, 5, 5, 5, 5, 5, 5, 5, 5, 1])).cuda()
trainloader, testloader        = get_loader(args.batch_size, args.size, 9, mask=True)


optimizer   = None
if args.optimizer == "sgd":
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
else:
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
            

for e in range(args.num_epochs):
    train_epoch(model, trainloader,testloader, optimizer, criterion, device, writer, e, args)
    scheduler.step()
    torch.save(model.state_dict(), 'runs/'+args.name+'/model.pth')



