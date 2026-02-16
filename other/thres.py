import numpy as np
from matplotlib import pyplot as plt


def thres(im1, im, db):
    preds_test_thresh = np.copy(im1)
    if db ==1:
        r, c = np.where(im >= 100)
        r1 = np.random.choice(r, int(len(r)/1.5))
        c1 = np.random.choice(c, int(len(c)/1.5))
    else:
        r, c = np.where(im >= 100)
        r1 = np.random.choice(r, int(len(r)/1.5))
        c1 = np.random.choice(c, int(len(c)/1.5))
    # r, c = np.where(preds_test_thresh == 0)
    # r2 = np.random.choice(r, 10000)
    # c2 = np.random.choice(c, 10000)
    preds_test_thresh[(r1, c1)] = im[(r1, c1)]
    img = np.where(preds_test_thresh > 0, 1, preds_test_thresh)
    # plt.imshow(img)
    return img