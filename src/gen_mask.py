from dataset import EyeSegmentDataset
import numpy as np
import cv2

import matplotlib.pyplot as plt

ds = EyeSegmentDataset(num_curves=9, transforms=None, file_path=True, mask=False)


for a in range(len(ds)):
    img, lab, path = ds[a]
    lab = np.array(lab, dtype=np.int)
    w, h = img.size
    print(w, h)
    print(lab.shape)
    seg_lab = np.zeros((w, h), dtype=np.int)

    for i in range(w):
        lab_id = 0
        # print(lab[i])
        j = 0
        while j < h:
            if lab_id == 9 or j < lab[i,lab_id]:
                seg_lab[i, j] = lab_id
            elif j == lab[i,lab_id]:
                while(lab_id < 9 and j== lab[i,lab_id]):
                    # print(j, lab_id)
                    lab_id += 1
                    # j+= 1
                seg_lab[i, j] = lab_id
            j += 1

    
    # matplotlib.image.imsave('seg_lab.png', seg_lab)
    # print(path)
    cv2.imwrite(path.replace('.bmp', '_mask.png'), seg_lab.transpose(1,0))

    # break


    