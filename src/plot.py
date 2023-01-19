import matplotlib.pyplot as plt



def plot_grid(imgs, labs, masks, rows=3, cols=3, lab_normalized=False):
    fig = plt.figure(figsize=(5*rows,5*cols), frameon=False)
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
    

def layer_plot(image, label):
    # print(image.shape, label.shape)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image)
    num_layers = label.shape[1]
    for i in range(num_layers):
        ax.plot(label[:,i])
    return fig

