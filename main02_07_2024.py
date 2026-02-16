import os
import cv2
import pandas as pd
from numpy import array as ar
from colorama import Fore, init
from pytictoc import TicToc
from scipy.ndimage import gaussian_filter1d
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, ReLU, Bidirectional, LSTM, Flatten, Dropout, Dense, Conv1D, MaxPooling1D
from tensorflow.python.keras.models import Model, Sequential
import argparse
from FCM import FCM
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from other import result, popup, Confusion_matrix
from other.norm_ import *  # Import all functions from norm_
from sklearn import svm
from matplotlib import pyplot as plt

def ful_analysis(db):

    def get_args():
        parser = argparse.ArgumentParser(
            description="Video Action Recognition.")
        # -----------------Algorithm-----------------
        parser.add_argument('-a', '--algorithm', default='FCM', choices=['FCM', 'EnFCM', 'MFCM'], type=str,
                            help="Choose a fuzzy c-means clustering algorithm. (FCM, EnFCM, MFCM)")
        parser.add_argument('--num_bit', default=8, type=int,
                            help="number of bits of input images")
        # -----------------Fundamental parameters-----------------
        parser.add_argument('-c', '--num_cluster', default='4', type=int,
                            help="Number of cluster")
        parser.add_argument('-m', '--fuzziness', default='2', type=int,
                            help="fuzziness degree")
        parser.add_argument('-i', '--max_iteration', default='5', type=int,
                            help="max number of iterations.")
        parser.add_argument('-e', '--epsilon', default='0.05', type=float,
                            help="threshold to check convergence.")
        # -----------------User parameters-----------------
        parser.add_argument('--plot_show', default=1, choices=[0, 1],
                            help="Show plot about result")
        parser.add_argument('--plot_save', default=1, choices=[0, 1],
                            help="Save plot about result")
        # -----------------Parametesr for MFCM/EnFCM-----------------
        parser.add_argument('-w', '--win_size', default='5', type=int,
                            help="Window size of MFCM/EnFCM algorithm")
        parser.add_argument('-n', '--neighbour_effect', default='3', type=float,
                            help="Effect factor of the graylevel which controls the influence extent of neighbouring pixels.")

        args = parser.parse_args()
        return args

    def extract(db):
        if db == 1:
            folder = '.\Dataset/UCF-101'
            label, jobs_ = [], []
            cond = 0
            for filename in os.listdir(folder):
                print('folder::::', filename)
                VI = []
                for video in os.listdir(os.path.join(folder, filename)):
                    VID1 = []
                    cap = cv2.VideoCapture(os.path.join(folder, filename, video))
                    if (cap.isOpened() == False):
                        print("Error opening video  file")
                    while (cap.isOpened()):
                        ret, frame = cap.read()
                        if ret is False:
                            cap.release()
                            break
                        VID1.append(frame)
                    fm = np.linspace(0, VID1.__len__() - 1, 3).round()
                    for i in fm:
                        VI=VID1[int(i)]
                        jobs_.append(VI)
                        label.append(cond)
                cond += 1
        elif db == 2:
            folder = '.\Dataset/HACS'
            label, jobs_ = [], []
            cond = 0
            for filename in os.listdir(folder):
                VI = []
                print('folder::::', filename)
                for video in os.listdir(os.path.join(folder, filename)):
                    VID1 = []
                    cap = cv2.VideoCapture(os.path.join(folder, filename, video))
                    if (cap.isOpened() == False):
                        print("Error opening video  file")
                    while (cap.isOpened()):
                        ret, frame = cap.read()
                        if ret is False:
                            cap.release()
                            break
                        VID1.append(frame)
                        # for i in range(0,VID1.__len__()):
                    fm = np.linspace(0, VID1.__len__() - 1, 5).round()
                    for i in fm:
                        VI = VID1[int(i)]
                        jobs_.append(VI)
                        label.append(cond)
                cond += 1
        return np.array(jobs_), np.array(label)

    def Preprocessing(db, Data, label):
        ln = len(np.unique(label))
        id = []
        # Loop through numbers 0 to 9
        for i in range(ln):
            # Find the indices where ln equals the current number i
            indices = np.where(label == i)[0]
            # Append the first 5 indices (or fewer if there are not enough occurrences) to the list
            id.extend(indices[:5])
        # Convert id to a numpy array
        id = np.array(id)
        data = Data
        filtered_signal,filtered_signal1,original_signal = [],[],[]
        sigma = 1.0
        i=0
        for sig in data:
            img = gaussian_filter1d(sig, sigma)
            filtered_signal1.append(img)
            if i in id:
                filtered_signal.append(img)
                original_signal.append(sig)
            i += 1
        filtered_signal1 = np.array(filtered_signal1)
        np.save(f'pre_evaluated/Filtered_Signal{db}', filtered_signal)
        np.save(f'pre_evaluated/Original_signal{db}', original_signal)
        return filtered_signal1

    def Segmentation(db,Data,label):
        def segmentation(seg_model, image, ln):
            ir = image
            if image.shape[-1] != 3:
                img = np.repeat(image[:, :, np.newaxis], 3, axis=-1)
            else:
                img = image
            input_shape = np.shape(image)
            seg = seg_model.predict(img[np.newaxis, :, :, :])
            print(f"Shape of seg model prediction: {seg.shape}")
            num_classes = seg.shape[-1]
            y_pred = resize(seg, image, axis=ln)
            return y_pred
        def unpooling2d(x, pool_size=(2, 2)):
            # Get the shape of the input tensor
            input_shape = tf.shape(x)
            h, w = input_shape[1], input_shape[2]

            # Perform upsampling
            upsampled = tf.image.resize(x, size=(h * pool_size[0], w * pool_size[1]), method='nearest')
            return upsampled
        # Custom loss function
        def custom_loss(y_true, y_pred, h, w):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)  # Avoid log(0)
            term1 = (-2 * y_true + (y_true ** 2) / (h * w)) * tf.math.log(y_pred)
            term2 = ((1 - y_true) + (y_true * (1 - y_true)) / (h * w)) * tf.math.log(1 - y_pred)
            loss = tf.reduce_mean(term1 - term2)
            return loss
        # SegNet model with Unpooling2D and custom loss function
        def imp_segnet(input_shape, num_classes, h, w):
            inputs = Input(shape=input_shape)

            # Encoder
            x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

            x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

            x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

            x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

            x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

            # Decoder
            x = unpooling2d(x, pool_size=(2, 2))
            x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = unpooling2d(x, pool_size=(2, 2))

            x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = unpooling2d(x, pool_size=(2, 2))

            x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = unpooling2d(x, pool_size=(2, 2))

            x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            outputs = Conv2D(num_classes, (1, 1), padding='same', activation='softmax')(x)
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam',
                          loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, h, w),
                          metrics=['accuracy'])
            return model
        def segnet(input_shape, num_classes):
            inputs = Input(shape=input_shape)
            # Encoder
            x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
            x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
            x = MaxPooling2D()(x)

            x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
            x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
            x = MaxPooling2D()(x)

            # Decoder
            x = UpSampling2D()(x)
            x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
            x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)

            x = UpSampling2D()(x)
            x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
            x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
            outputs = Conv2D(num_classes, (1, 1), activation='softmax')(x)
            model = Model(inputs=inputs, outputs=outputs)
            return model
        # Assuming you have `db` defined elsewhere
        input_shape = np.shape(Data[0])  # Adjust input shape according to your data
        # input_shape = cv2.resize(input_shape, (256, 256))
        ln = len(np.unique(label))
        id = []
        # Loop through numbers 0 to 9
        for i in range(ln):
            # Find the indices where ln equals the current number i
            indices = np.where(label == i)[0]
            # Append the first 5 indices (or fewer if there are not enough occurrences) to the list
            id.extend(indices[:5])
        # Convert id to a numpy array
        id = np.array(id)

        num_classes = ln  # Adjust based on the number of classes in your segmentation task
        seg_model = segnet(input_shape, num_classes)
        seg_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        impseg_model = imp_segnet(input_shape, num_classes, input_shape[0], input_shape[1])

        SEG, impS, S, fcmSeg, Kmeans, impSeg, conSeg = [], [], [], [], [], [], []
        args = get_args()
        i = 0
        for sig in Data:
            pre_processed_image = sig
            # Imp Segnet Segmentation
            segmented_image = segmentation(impseg_model, pre_processed_image, 1)
            impS.append(segmented_image)
            if i in id:
                impSeg.append(segmented_image)
            # conv Segnet Segmentation
            segmented_image = segmentation(seg_model, pre_processed_image, 0)
            S.append(segmented_image)
            if i in id:
                conSeg.append(segmented_image)
            # FCM clustering
            img1 = cv2.cvtColor(sig, cv2.COLOR_BGR2GRAY)
            img1 = cv2.resize(img1, (256, 256))
            args = get_args()
            cluster = FCM(img1, image_bit=args.num_bit, n_clusters=5, m=args.fuzziness,
                          epsilon=args.epsilon,
                          max_iter=args.max_iteration)
            cluster.form_clusters()
            if i in id:
                fcmSeg.append(cluster.result)

            # Kmeans Clustering
            img1 = cv2.cvtColor(sig, cv2.COLOR_BGR2GRAY)
            img1=cv2.resize(img1, (256, 256))
            img1 = img1.flatten().reshape(-1, 1)
            kmeans = KMeans(n_clusters=5)
            a = kmeans.fit(img1)
            centroids = kmeans.cluster_centers_
            labels = kmeans.labels_
            lab = labels.reshape(256, 256)
            if i in id:
                Kmeans.append(lab)
            i += 1
        np.save(f"pre_evaluated/imp_segmented_images{db}.npy", np.array(impSeg))
        np.save(f"pre_evaluated/con_segmented_images{db}.npy", np.array(conSeg))
        np.save(f"pre_evaluated/FCM_segmented_images{db}.npy", np.array(fcmSeg))
        np.save(f"pre_evaluated/KMeans_segmented_images{db}.npy", np.array(Kmeans))
        return np.array(impS), np.array(S)

    def FeatureExtraction(db,Data,conData):
        def imp_phog(image, num_bins=9, cell_size=(8, 8), block_size=(2, 2), pyramid_levels=3):
            if image.dtype != np.uint8:
                image = cv2.convertScaleAbs(image)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gradient_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
            gradient_magnitude = np.abs(gradient_x) + np.abs(gradient_y)
            gradient_orientation = np.arctan2(gradient_y, gradient_x)
            bin_width = 360.0 / num_bins
            quantized_orientation = np.floor(gradient_orientation / bin_width).astype(int)
            pyramid = [image]
            for level in range(1, pyramid_levels):
                resized_image = cv2.resize(pyramid[level - 1], (0, 0), fx=0.5, fy=0.5)
                pyramid.append(resized_image)
            phog_features = []
            for level_image in pyramid:
                level_features = []
                for y in range(0, level_image.shape[0] - cell_size[1] + 1, cell_size[1]):
                    for x in range(0, level_image.shape[1] - cell_size[0] + 1, cell_size[0]):
                        cell_magnitude = gradient_magnitude[y:y + cell_size[1], x:x + cell_size[0]]
                        cell_orientation = quantized_orientation[y:y + cell_size[1], x:x + cell_size[0]]
                        hist, _ = np.histogram(cell_orientation, bins=num_bins, range=(0, num_bins),
                                               weights=cell_magnitude)
                        level_features.extend(hist)
                level_features = np.array(level_features, dtype=np.float64)
                level_features /= np.linalg.norm(level_features + 1e-6)
                phog_features.extend(level_features)
            return np.array(phog_features)

        def conventional_phog(image, num_bins=9, cell_size=(8, 8), block_size=(2, 2), pyramid_levels=3):
            if image.dtype != np.uint8:
                image = cv2.convertScaleAbs(image)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gradient_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
            gradient_orientation = np.arctan2(gradient_y, gradient_x) * (180.0 / np.pi) % 360.0
            bin_width = 360.0 / num_bins
            quantized_orientation = np.floor(gradient_orientation / bin_width).astype(int)
            pyramid = [image]
            for level in range(1, pyramid_levels):
                resized_image = cv2.resize(pyramid[level - 1], (0, 0), fx=0.5, fy=0.5)
                pyramid.append(resized_image)
            phog_features = []
            for level_image in pyramid:
                level_features = []
                for y in range(0, level_image.shape[0] - cell_size[1] + 1, cell_size[1]):
                    for x in range(0, level_image.shape[1] - cell_size[0] + 1, cell_size[0]):
                        cell_magnitude = gradient_magnitude[y:y + cell_size[1], x:x + cell_size[0]]
                        cell_orientation = quantized_orientation[y:y + cell_size[1], x:x + cell_size[0]]
                        hist, _ = np.histogram(cell_orientation, bins=num_bins, range=(0, num_bins),
                                               weights=cell_magnitude)
                        level_features.extend(hist)
                level_features = np.array(level_features, dtype=np.float64)
                level_features /= np.linalg.norm(level_features + 1e-6)
                phog_features.extend(level_features)
            return np.array(phog_features)

        def shape_feat(image):
            if image.dtype != np.uint8:
                image = cv2.convertScaleAbs(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
            contours = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if not contours:
                return np.zeros(50)
            cont_shape1 = max((contours[i].shape) for i in range(len(contours)))
            for i in range(len(contours)):
                if contours[i].shape == cont_shape1:
                    iter = i
            cnt = contours[iter]
            cnt1 = np.concatenate((cnt, cnt), axis=0)
            M = cv2.moments(cnt1)
            if M['m00'] == 0:
                return np.zeros(50)
            moment_values = np.array(list(M.values()))
            shape_features = [moment_values]
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            area = cv2.contourArea(cnt)
            shape_features.append(area)
            perimeter = cv2.arcLength(cnt, True)
            shape_features.append(perimeter)
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            shape_features.append(epsilon)
            hull = cv2.convexHull(cnt)
            k = cv2.isContourConvex(cnt)
            shape_features.append(int(k))
            ss = np.hstack(shape_features)
            ssf = np.histogram(ss, bins=50)[0]
            return ssf

        def color_feat_(img):
            img = img.astype('uint8')
            if len(img.shape) == 2:
                img = cv2.merge([img, img, img])
            elif len(img.shape) == 3 and img.shape[2] == 1:
                img = cv2.merge([img[:, :, 0], img[:, :, 0], img[:, :, 0]])
            color_f = []
            L, A, B = cv2.split(img)
            histr_A_bin4 = cv2.calcHist([A], [0], None, [4], [0, 256])
            histr_A_bin8 = cv2.calcHist([A], [0], None, [8], [0, 256])
            histr_A_bin16 = cv2.calcHist([A], [0], None, [16], [0, 256])
            histr_A_bin32 = cv2.calcHist([A], [0], None, [32], [0, 256])

            color_f.append(histr_A_bin4)
            color_f.append(histr_A_bin8)
            color_f.append(histr_A_bin16)
            color_f.append(histr_A_bin32)

            histr_B_bin4 = cv2.calcHist([B], [0], None, [4], [0, 256])
            histr_B_bin8 = cv2.calcHist([B], [0], None, [8], [0, 256])
            histr_B_bin16 = cv2.calcHist([B], [0], None, [16], [0, 256])
            histr_B_bin32 = cv2.calcHist([B], [0], None, [32], [0, 256])

            color_f.append(histr_B_bin4)
            color_f.append(histr_B_bin8)
            color_f.append(histr_B_bin16)
            color_f.append(histr_B_bin32)

            aa = np.array(color_f)
            clr_feat = np.concatenate((aa[0], aa[1], aa[3], aa[4], aa[5], aa[6], aa[7]), axis=0)
            clr_feat_ = np.histogram(clr_feat, bins=50)[0]
            return clr_feat_
        # Load data
        final_features, pfinal_features = [], []
        for img in Data:
            iphog = imp_phog(img)
            iphog_ = np.histogram(iphog, bins=100)
            cphog = conventional_phog(img)
            cphog_ = np.histogram(cphog, bins=100)
            sfeat = shape_feat(img)
            cfeat = color_feat_(img)
            # Concatenate features
            pfinal_features.append(np.concatenate((iphog_[0], sfeat, cfeat)))
            final_features.append(np.concatenate((cphog_[0], sfeat, cfeat)))
        np.save(f"pre_evaluated/Feature{db}.npy", np.array(pfinal_features))
        np.save(f"pre_evaluated/conFeature{db}.npy", np.array(final_features))
        csfinal_features = []

        for img in conData:
            iphog = imp_phog(img)
            iphog_ = np.histogram(iphog, bins=100)
            sfeat = shape_feat(img)
            cfeat = color_feat_(img)
            # Concatenate features
            csfinal_features.append(np.concatenate((iphog_[0], sfeat, cfeat)))
        np.save(f"pre_evaluated/conSegFeature{db}.npy", np.array(csfinal_features))
        # return np.array(pfinal_features), np.array(final_features), np.array(csfinal_features)

    def data_Augmentation(data, lab):
        ln = len(set(lab))
        aug_data, aug_lab = [], []
        cnt = [sum(lab == i) for i in range(ln)]
        for i in range(ln):
            a = max(cnt)
            ag_data = np.zeros([2*a+(a-cnt[i]), data.shape[1]])   # 2*a+(a-cnt[i]) to balance the label
            ag_lab = np.zeros([2*a+(a-cnt[i])])
            Datas = data[np.where(lab == i)]
            Label = lab[np.where(lab == i)]
            for j in range(Datas.shape[1]):
                lb = np.min(Datas[:, j])
                ub = np.max(Datas[:, j])
                ag_data[:, j] = np.random.uniform(lb, ub, ag_data.shape[0])
            ag_lab[:] = i
            aug_data.extend(np.concatenate((Datas, ag_data), axis=0))
            aug_lab.extend(np.concatenate((Label, ag_lab), axis=0))
        augment_data, augment_lab = np.array(aug_data), np.array(aug_lab).astype('int')
        return augment_data, augment_lab

    def DNN():
        t = TicToc()
        t.tic()
        model = Sequential()
        model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
        ln = len(np.unique(Y_train))
        model.add(Dense(ln, activation='softmax'))
        model.add(Dense(ln, activation='softmax'))
        model.add(Dense(ln, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        X_tr = X_train.reshape(X_train.shape[0], X_train.shape[1])
        X_tst = X_test.reshape(X_test.shape[0], X_test.shape[1])
        model.fit(X_tr, Y_train, epochs=1, batch_size=100, verbose=1)
        predict = array(np.argmax(model.predict(X_tst), axis=-1), axis=0)
        metric = Confusion_matrix.multi_confu_matrix(Y_test, predict)
        t.toc()
        return metric[0], metric[1], predict, t.tocvalue()

    def Lstm():
        t = TicToc()
        t.tic()
        lstm_X_train = X_train.reshape(-1, X_train.shape[1], 1)
        lstm_X_test = X_test.reshape((-1, X_test.shape[1], 1))
        inp_shp = lstm_X_train.shape[1:]
        model = Sequential()
        model.add(LSTM(128, input_shape=inp_shp, return_sequences=True))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dropout(0.2))
        # number of features on the output
        model.add(Dropout(0.2))
        model.add(Dense(ln, activation='sigmoid'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        model.fit(lstm_X_train, np.array(Y_train).transpose(), batch_size=100, epochs=1)
        predict = array(np.argmax(model.predict(lstm_X_test), axis=1), axis=0)
        metric = Confusion_matrix.multi_confu_matrix(Y_test, predict)
        t.toc()
        return metric[0], metric[1], predict, t.tocvalue()

    def SVM():
        t = TicToc()
        t.tic()
        clf = svm.SVC(C=2)  # C=0.01, kernel='linear',  tol=3)
        train_X = X_train.reshape(-1, X_train.shape[1], 1)
        test_X = X_test.reshape(-1, X_train.shape[1], 1)
        train_X = train_X.astype('float32')
        test_X = test_X.astype('float32')
        clf.fit(X_train.reshape(X_train.shape[0], X_train.shape[1]), Y_train)
        Pred = array(clf.predict(test_X.reshape(X_test.shape[0], X_test.shape[1])), axis=0)  # 0:Overcast, 2:Mild
        metric = Confusion_matrix.multi_confu_matrix(Y_test, Pred)
        t.toc()
        return metric[0], metric[1], Pred, t.tocvalue()

    def CNN():
        t = TicToc()
        t.tic()
        cnn_X_train = X_train.reshape(-1, X_train.shape[1], 1)
        cnn_X_test = X_train.reshape(-1, X_test.shape[1], 1)
        inp_shp = cnn_X_train.shape[1:]
        model = Sequential()
        model.add(Conv1D(64, (1,), padding='valid', input_shape=inp_shp, activation='relu'))
        model.add(MaxPooling1D(pool_size=(1,)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(ln, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(cnn_X_train, np.array(Y_train).transpose(), epochs=1, batch_size=100, verbose=1)
        predict = array(np.argmax(model.predict(cnn_X_test), axis=1), axis=0)
        metric = Confusion_matrix.multi_confu_matrix(Y_test, predict)
        t.toc()
        return metric[0], metric[1], predict, t.tocvalue()

    def Lnet():
        from LinkNet import linknet
        t = TicToc()
        t.tic()
        y_pred = array(linknet(X_train, Y_train, X_test, Y_test), axis=0)
        metric =Confusion_matrix.multi_confu_matrix(Y_test, y_pred)
        t.toc()
        return metric[0], metric[1], y_pred, t.tocvalue()

    def BiLstm():
        t = TicToc()
        t.tic()
        lstm_X_test = X_test.reshape((-1, X_test.shape[1], 1))
        lstm_X_train = X_train.reshape(-1, X_train.shape[1], 1)
        model = Sequential()
        inp_shp = lstm_X_train.shape[1:]
        model.add(Bidirectional(LSTM(128, input_shape= inp_shp, activation='softmax')))
        model.add(Dense(ln, activation='sigmoid'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(lstm_X_train, np.array(Y_train).transpose(), epochs=1, batch_size=100, verbose=1)
        model.save(f'pre_evaluated/Bilstm_model{lpstr}.h5')
        predict = array(np.argmax(model.predict(lstm_X_test), axis=1), axis=0)
        metric = Confusion_matrix.multi_confu_matrix(Y_test, predict)
        t.toc()
        return metric[0], metric[1], predict, t.tocvalue()

    def KNN():
        t = TicToc()
        t.tic()
        from sklearn import neighbors
        model_ = neighbors.KNeighborsClassifier()
        model_.fit(X_train, Y_train)
        y_pred = array(model_.predict(X_test), axis=0)
        metric = Confusion_matrix.multi_confu_matrix(Y_test, y_pred)
        t.toc()
        return metric[0], metric[1], y_pred, t.tocvalue()

    # Define the custom loss function
    def custom_loss(beta=0.5):
        def loss(y_true, y_pred):
            # Clip predictions to avoid log(0) errors
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            # Compute individual terms of the loss function
            pp = y_true * y_pred
            pp_beta = beta * (1 - y_true) * y_pred
            pp_one_minus_beta = (1 - beta) * y_true * (1 - y_pred)
            numerator = pp - y_true * tf.math.log(y_pred)
            denominator = pp + pp_beta + pp_one_minus_beta
            # Compute the final loss
            loss_value = numerator / (denominator + 1e-7)  # Added a small constant to avoid division by zero
            return tf.reduce_mean(loss_value)
        return loss

    # Define the BiLSTM model
    def bilstm_model(input_shape, num_classes, dropout_rate=0.5):
        inputs = Input(shape=input_shape)
        # BiLSTM layer
        x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        x = BatchNormalization()(x)
        # Flatten layer
        x = Flatten()(x)
        # Dropout layer
        x = Dropout(dropout_rate)(x)
        # Dense layer
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        # Dropout layer
        x = Dropout(dropout_rate)(x)
        # Output layer with softmax
        outputs = Dense(num_classes, activation='softmax')(x)
        # Create model
        model = Model(inputs, outputs)
        # Compile model
        model.compile(optimizer='adam', loss=custom_loss(beta=0.5), metrics=['accuracy'])
        return model

    def hybrid():
        t = TicToc()
        t.tic()
        # Bilstm
        # Reshape the data
        lstm_X_train = X_train.reshape(-1, X_train.shape[1], 1)
        lstm_X_test = X_test.reshape(-1, X_test.shape[1], 1)
        # Convert Y_train to one-hot encoding
        num_classes = len(np.unique(Y_train))
        # Define the input shape
        input_shape = lstm_X_train.shape[1:]
        # Create the model
        model = bilstm_model(input_shape, num_classes)
        # Train the model
        model.fit(lstm_X_train, np.array(Y_train).transpose(), epochs=100, batch_size=50, verbose=1)
        model.save(f'pre_evaluated/Bilstm_model{lpstr}.h5')
        y_pred1 = array(np.argmax(model.predict(lstm_X_test), axis=1), axis=1)
        from LinkNet import linknet
        y_pred2 = array(linknet(X_train, Y_train, X_test, Y_test), axis=1)
        # Stack the prediction arrays
        predictions = np.vstack((y_pred1, y_pred2))
        # Compute the mean of the stacked arrays along axis 0
        mean_predictions = np.mean(predictions, axis=0)
        # Round the mean predictions to get class labels
        predict = np.round(mean_predictions).astype(int)
        metric = Confusion_matrix.multi_confu_matrix(Y_test, predict)
        t.toc()
        return metric[0], metric[1], predict, t.tocvalue()

    def stat_analysis(xx):
        from numpy import mean, median, std, min, max
        mn = mean(xx, axis=0).reshape(-1, 1)
        mdn = median(xx, axis=0).reshape(-1, 1)
        std_dev = std(xx, axis=0).reshape(-1, 1)
        mi = min(xx, axis=0).reshape(-1, 1)
        mx = max(xx, axis=0).reshape(-1, 1)
        return np.concatenate((mn, mdn, std_dev, mi, mx), axis=1)


    # # Read the Dataset
    images, label = extract(db)

    # # Preprocess the Dataset
    preimage = Preprocessing(db, images, label)

    # #  Segmentation
    segimage, csegimage = Segmentation(db, preimage, label)

    # # FeatureExtraction
    Feature, conFeature, conSegFeature = FeatureExtraction(db, segimage, csegimage)
    Data = Feature

    # # # Data Augmentation
    augment_data, augment_lab = data_Augmentation(Data, label)

    run =False
    if run:
        learn_percent, learning_percentage = [0.6, 0.7, 0.8, 0.9], ['60', '70', '80', '90']
        ln = len(set(label))
        true_lab, pred_label = [], []
        for lp, lpstr in zip(learn_percent, learning_percentage):
            X_train, X_test, Y_train, Y_test, = train_test_split(augment_data, augment_lab, train_size=lp,
                                                                 random_state=0)
            np.save('pre_evaluated/Y_test', Y_test);
            true_lab.append(Y_test);
            np.save('pre_evaluated/lp', lp)
            result = [Lstm(),CNN(), DNN(), SVM(), KNN(), Lnet(), BiLstm(),hybrid()]
            perf = ar([result[i][0] for i in range(len(result))])
            TPTN = ar([result[i][1] for i in range(len(result))])
            pred_label = np.array([result[i][2] for i in range(len(result))])
            comtim = np.array([result[i][3] for i in range(len(result))])
            np.save(f'pre_evaluated/pred_label{lpstr}', pred_label)
            clmn = ['LSTM', 'CNN', 'DNN', 'SVM', 'KNN', 'LinkNet', 'BiLSTM', 'PROP']
            indx1 = ['TP', 'TN', 'FP', 'FN']
            indx = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f_measure', 'mcc', 'npv', 'fpr', 'fnr','fdr']
            indx2 = ['computational time']
            comtim = np.array(comtim)
            globals()['df' + lpstr] = pd.DataFrame(perf.transpose(), columns=clmn, index=indx)
            globals()['df1' + lpstr] = pd.DataFrame(TPTN.transpose(), columns=clmn, index=indx1)
            globals()['df2' + lpstr] = pd.DataFrame(comtim.reshape(1, -1), columns=clmn, index=indx2)

        key = ['60', '70', '80', '90']
        frames = [df60, df70, df80, df90]
        frames1 = [df160, df170, df180, df190]
        frames2 = [df260, df270, df280, df290]
        df = pd.concat(frames, keys=key, axis=0)
        df1 = pd.concat(frames1, keys=key, axis=0)
        df2 = pd.concat(frames2, keys=key, axis=0)

        stat = df.loc[(key, ['accuracy']), :].values
        stat = stat_analysis(stat).transpose()
        df_ = pd.DataFrame(stat, ['Mean', 'Median', 'Std-Dev', 'Min', 'Max'],
                               ['LSTM', 'CNN', 'DNN', 'SVM', 'KNN', 'LinkNet', 'BiLSTM', 'PROP'])

        df.to_csv(f'pre_evaluated/Comparision{db}.csv')
        df1.to_csv(f'pre_evaluated/TPTN_result{db}.csv')
        df2.to_csv(f'pre_evaluated/time_result{db}.csv')
        np.save(f'pre_evaluated/truelabel{db}', true_lab)
        df_.to_csv(f'pre_evaluated/statistics analysis{db}.csv')

    run = False
    if run:
        # # Ablation_Study
        from comparision import Ablation_Study, image_res
        Ablation_Study(db, Feature, conFeature, conSegFeature, segimage)
        image_res(db)

    # --------Plot Results-----------------
    # Read the CSV file into a DataFrame
    df = pd.read_csv(f'pre_evaluated/Comp{db}.csv')
    x = pd.read_csv(f'pre_evaluated/Comparision{db}.csv', header=0, index_col=1,
                    error_bad_lines=False).iloc.obj['PROP']
    x = x.values
    com = df
    comp = np.concatenate((com,  x.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(comp, columns=['Training Data',  'Metrics',  'Proposed+confeature', 'Proposed+conSegmentation', 'Proposed+conClassifier', 'Proposed+withoutFeature', 'Proposed'])
    df.to_csv(f'pre_evaluated/Ablation Study{db}.csv')

# -----Plot Results---------#
    column = ['LSTM', 'CNN', 'DNN', 'SVM', 'KNN', 'LinkNet', 'BiLSTM', 'PROP']
    plot_result = pd.read_csv(f'pre_evaluated/Comparision{db}.csv', index_col=[0, 1])
    indx = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f_measure', 'mcc', 'npv', 'fpr', 'fnr', 'fdr']

    avg = plot_result.loc[90, :]
    avg.reset_index(drop=True, level=0)
    avg.to_csv(f'Results/Performance Analysis{db}.csv')
    print('\n\t', Fore.LIGHTBLUE_EX + f'Performance Analysis{db}')
    print(avg.to_markdown())

    print('\n\t', Fore.LIGHTBLUE_EX + f'Ablation Study{db}')
    anal = pd.read_csv(f'pre_evaluated/Ablation Study{db}.csv', index_col=[0], header=[0])
    print(pd.read_csv(f'pre_evaluated/Ablation Study{db}.csv', index_col=[0], header=[0]).to_markdown())
    anal.to_csv(f'Results/Ablation Study{db}.csv')

    print('\n\t', Fore.LIGHTBLUE_EX + f'Statistics analysis{db}')
    state = pd.read_csv(f'pre_evaluated/statistics analysis{db}.csv', header=0, names=column)
    print(pd.read_csv(f'pre_evaluated/statistics analysis{db}.csv', header=0, names=column).to_markdown())
    state.to_csv(f'Results/Statistics analysis{db}.csv')

    # Analysis plot
    for idx, jj in enumerate(indx):
        colors = ['#669999', 'lime', '#0040ff', '#ffff00', '#4d4d4d', '#9900ff', '#00ffff', '#ff0000']
        new_ = plot_result.loc[([60, 70, 80, 90], [jj]), :]
        new_.reset_index(drop=True, level=1, inplace=True)
        new = new_.values
        br1 = np.arange(4)
        plt.style.use('grayscale')
        plt.figure(figsize=(12, 7))
        for i in range(new.shape[1]):
            plt.bar(br1, new[:, i], color=colors[i], width=0.085,
                edgecolor='k', label=column[i])
            br1 = [x + 0.085 for x in br1]
        plt.subplots_adjust(bottom=0.2)
        plt.grid(color='g', linestyle=':', linewidth=0.9)
        plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), fontsize=12, ncol=5)
        plt.xlabel('Training Data (%)', weight='bold', size=17)
        plt.ylabel(jj.upper(), weight='bold', size=17)
        plt.xticks([r for r in range(4)],
               ['60', '70', '80', '90'])
        plt.savefig(f'Results/Dataset{db}-{jj.upper()}.png', dpi=800)
    plt.show()

init(autoreset=True)
popup.popup(ful_analysis, result.result)