import torch
from plot import plot_grid
import matplotlib.pyplot as plt
import numpy as np


def train_epoch(net, trainloader, optimizer, criterion, device, writer, epoch_no, args):

    i = epoch_no*len(trainloader)*args.batch_size

    output_pts = None
    lab_reshaped = None
    
    for images, labels in trainloader:
        i += images.shape[0]
        labels = labels.type(torch.FloatTensor)
        images = images.type(torch.FloatTensor)
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
      
        
        output_pts = net(images)

        loss = criterion(output_pts, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        writer.add_scalar('Loss', loss.item(), i)
        
    print('[Epoch %d] loss: %.3f' %
                    (epoch_no + 1, loss.item()))
    out_reshaped = output_pts.cpu().data.numpy() 
    lab_reshaped = labels.cpu().data.numpy()
    out_lab = np.concatenate((out_reshaped, lab_reshaped), 2)
        
    fig = plot_grid(images.cpu().data.numpy(), out_lab)
    writer.add_figure('predictions', fig, i)
    writer.flush()
    plt.close()
