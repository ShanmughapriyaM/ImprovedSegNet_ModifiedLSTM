import warnings, sys, os

import pandas as pd

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import tensorflow
simplefilter("ignore", category=ConvergenceWarning)
from tensorflow.python.keras.models import Model
import numpy as np
import cv2
from sklearn.metrics import jaccard_score

def dice_coefficient(seg, gt):
    # Assume binary images, 1 for foreground and 0 for background
    intersection = np.sum(seg[gt == 1])
    return (2. * intersection) / (np.sum(seg) + np.sum(gt))

def rgb_to_binary(image, threshold_value=128):
    if image.ndim == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert RGB to Grayscale
    else:
        gray_image = image  # Image is already grayscale
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)  # Binarize the image
    binary_image = (binary_image > 0).astype(np.uint8)  # Ensure binary format (0 and 1)
    return binary_image

def dice_score(db):
    Org_img = np.load(f'pre_evaluated/Original_signal{db}.npy')
    pseg_img = np.load(f'pre_evaluated/imp_segmented_images{db}.npy')
    cseg_img = np.load(f'pre_evaluated/con_segmented_images{db}.npy')
    fseg_img = np.load(f'pre_evaluated/FCM_segmented_images{db}.npy')
    kseg_img = np.load(f'pre_evaluated/KMeans_segmented_images{db}.npy')

    dice_scores = []
    # Loop through the first 5 images
    for i in range(10):
        # Resize and binarize the original image
        img = cv2.resize(Org_img[i], (256, 256))
        orgimg = rgb_to_binary(img)

        # Resize and binarize the predicted segmented image
        pimg = cv2.resize(pseg_img[i], (256, 256))
        pimg_binary = rgb_to_binary(pimg[:, :, 0] if pimg.ndim == 3 else pimg)
        # Calculate Dice score for predicted segmented image
        prop_dice = dice_coefficient(pimg_binary, orgimg)

        # Resize and binarize the convolutional segmented image
        cimg = cv2.resize(cseg_img[i], (256, 256))
        cimg_binary = rgb_to_binary(cimg[:, :, 0] if cimg.ndim == 3 else cimg)
        # Calculate Dice score for convolutional segmented image
        conv_dice = dice_coefficient(cimg_binary, orgimg)

        # Resize and binarize the KMeans segmented image directly
        kmeans_img = cv2.resize(kseg_img[i], (256, 256))
        # Calculate Dice score for KMeans segmented image
        kmeans_dice = dice_coefficient(kmeans_img, orgimg)

        # Resize and binarize the FCM segmented image directly
        fcm_img = cv2.resize(fseg_img[i], (256, 256))
        # Calculate Dice score for FCM segmented image
        fcm_dice = dice_coefficient(fcm_img, orgimg)

        dice_scores.append([prop_dice, conv_dice, kmeans_dice, fcm_dice])
    return np.array(dice_scores)

def jacc_Score(db):
    Org_img = np.load(f'pre_evaluated/frame{db}.npy')
    pseg_img = np.load(f'pre_evaluated/imp_segmented_images{db}.npy')
    cseg_img = np.load(f'pre_evaluated/con_segmented_images{db}.npy')
    fseg_img = np.load(f'pre_evaluated/FCM_segmented_images{db}.npy')
    kseg_img = np.load(f'pre_evaluated/KMeans_segmented_images{db}.npy')

    jaccard_scores = []
    # Loop through the first 5 images
    for i in range(10):
        # Resize and binarize the original image
        img = cv2.resize(Org_img[i], (256, 256))
        orgimg = rgb_to_binary(img)

        # Resize and binarize the predicted segmented image
        pimg = cv2.resize(pseg_img[i], (256, 256))
        pimg_binary = rgb_to_binary(pimg[:, :, 0] if pimg.ndim == 3 else pimg)
        # Calculate Jaccard score for predicted segmented image
        prop_jaccard = jaccard_score(orgimg.flatten(), pimg_binary.flatten(), average='binary')

        # Resize and binarize the convolutional segmented image
        cimg = cv2.resize(cseg_img[i], (256, 256))
        cimg_binary = rgb_to_binary(cimg[:, :, 0] if cimg.ndim == 3 else cimg)
        # Calculate Jaccard score for convolutional segmented image
        conv_jaccard = jaccard_score(orgimg.flatten(), cimg_binary.flatten(), average='binary')

        # Resize and binarize the KMeans segmented image directly
        kmeans_img = cv2.resize(kseg_img[i], (256, 256))
        # Calculate Jaccard score for KMeans segmented image
        kmeans_jaccard = jaccard_score(orgimg.flatten(), kmeans_img.flatten(), average='micro')

        # Resize and binarize the FCM segmented image directly
        fcm_img = cv2.resize(fseg_img[i], (256, 256))
        # Calculate Jaccard score for FCM segmented image
        fcm_jaccard = jaccard_score(orgimg.flatten(), fcm_img.flatten(), average='micro')

        jaccard_scores.append([prop_jaccard, conv_jaccard, kmeans_jaccard, fcm_jaccard])
    return np.array(jaccard_scores)

def imagecomparision(db):

    # Call the functions
    dice = dice_score(db)
    jacc = jacc_Score(db)
    score = [np.mean(dice, axis=0), np.mean(jacc, axis=0)]
    columns = ['metrics', 'ImpSegNet', 'Conv_SegNet', 'KMeans', 'FCM']
    rows = ['Dice', 'Jaccard']
    # Create DataFrame
    df = pd.DataFrame(score, columns=columns[1:])
    # Insert 'metrics' column
    df.insert(0, 'metrics', rows)
    df.to_csv(f'pre_evaluated/Dice-Jaccard{db}.csv', index=False)
