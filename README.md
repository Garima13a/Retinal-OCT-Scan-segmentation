# Retinal-OCT-Scan-segmentation
We have aimed to create a retinal OCT scan segmentation system that predicts all ten layer segments
present on retina. Our proof-of-concept study with a DeepLabV3 ResNet50 demonstrated convergence
of the model without overfitting/underfiting the data. We compared FCN ResNet50 ,FCN ResNet101,
DeepLabV3 ResNet50, DeepLabV3 ResNet101 (all with both Adam and SGD optimizers) to get the
best possible segmentation on OCT scan. Our deep learning system was trained and verified on imaging
data from two OCT devices (Handheld OCT and Table mounted OCT device), resulting in a device-
agnostic system.

'''
qsub main.sh
'''
