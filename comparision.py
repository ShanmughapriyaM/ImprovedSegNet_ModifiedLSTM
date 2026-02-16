import os
from other.norm_ import *
from sklearn.model_selection import train_test_split
import pandas as pd
from other.thres import *
from DICE_JACCARD import imagecomparision
import tensorflow as tf
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, ReLU, Bidirectional, LSTM, Flatten, Dropout, Dense, Conv1D, MaxPooling1D
from tensorflow.python.keras.models import Model, Sequential

def image_res(db):
    # Load data
    image = np.load(f'pre_evaluated/Original_signal{db}.npy')
    pre_image = np.load(f'pre_evaluated/Filtered_Signal{db}.npy')
    seg_image = np.load(f'pre_evaluated/imp_segmented_images{db}.npy')
    cseg_image = np.load(f'pre_evaluated/con_segmented_images{db}.npy')
    fseg_image = np.load(f'pre_evaluated/FCM_segmented_images{db}.npy')
    kseg_image = np.load(f'pre_evaluated/KMeans_segmented_images{db}.npy')
    label = np.load(f'pre_evaluated/Label{db}.npy')

    Sample_sig_idx = [1, 6, 11]

    filters = [
        ('Sample Image', image, 'Sample_image'),
        ('Gaussian Filtering', pre_image, 'Gaussian_Filtering_Sample_image'),
        ('ImpSegNet Segmentation', seg_image, 'ImpSegNet_Segmentation_Sample_image'),
        ('conSegNet Segmentation', cseg_image, 'conSegNet_Segmentation_Sample_image'),
        ('FCM Segmentation', fseg_image, 'FCM_Segmentation_Sample_image'),
        ('KMeans Segmentation', kseg_image, 'KMeans_Segmentation_Sample_image')
    ]

    # Create results directory if it doesn't exist
    result_dir = f'Results/image{db}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # Generate plots
    for i, idx in enumerate(Sample_sig_idx):
        for title, data, filename in filters:
            plt.figure()
            plt.title(title)
            img = data[idx]
            if title == 'ImpSegNet Segmentation' or title == 'conSegNet Segmentation':
                plt.imshow(img[:,:,0].astype('uint8'))
            else:
                plt.imshow(img.astype('uint8'))
            plt.savefig(f'{result_dir}/{filename}_{i + 1}.png')
            plt.close()

    imagecomparision(db)



def hybrid_classifier(X_train, Y_train, X_test, Y_test):
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
    return metric[0], metric[1], predict

def Ablation_Study(db,Feature,conFeature,conSegFeature,imp_segmented_images):
    def data_Augmentation(data, lab):
        ln = len(set(lab))
        aug_data, aug_lab = [], []
        cnt = [sum(lab == i) for i in range(ln)]
        for i in range(ln):
            a = max(cnt)
            ag_data = np.zeros([2 * a + (a - cnt[i]), data.shape[1]])  # 2*a+(a-cnt[i]) to balance the label
            ag_lab = np.zeros([2 * a + (a - cnt[i])])
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

    # Load saved data
    cfector = conFeature
    csfector = conSegFeature
    ccfector = Feature
    wofector = imp_segmented_images
    nan_mask = np.isnan(wofector)
    wofector[nan_mask] = -1
    wofector = (wofector - np.min(wofector)) / (np.max(wofector) - np.min(wofector))
    label = np.load(f'pre_evaluated/Label{db}.npy')
    run = False
    if run:
        learn_percent = [0.6, 0.7, 0.8, 0.9]
        learning_percentage = ['60', '70', '80', '90']
        ln = len(set(label))
        for lp, lpstr in zip(learn_percent, learning_percentage):
            results = []
            for feature_set in [cfector, csfector, ccfector,  wofector]:
                data, lab = data_Augmentation(feature_set, label)
                X_train,X_test, Y_train, Y_test = train_test_split(data, lab, train_size=lp, random_state=0)
                np.save('pre_evaluated/Y_test', Y_test);
                result = hybrid_classifier(X_train, Y_train, X_test, Y_test)
                results.append(result)

            comp = np.array([result[0] for result in results])
            TPTN = np.array([result[1] for result in results])

            clmn = ['Proposed+confeature', 'Proposed+conSegmentation', 'Proposed+conClassifier', 'Proposed+withoutFeature']
            indx1 = ['TP', 'TN', 'FP', 'FN']
            indx = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f_measure', 'mcc', 'npv', 'fpr', 'fnr',
                    'fdr']
            globals()[f'df{lpstr}'] = pd.DataFrame(comp.T, columns=clmn, index=indx)
            globals()[f'df1{lpstr}'] = pd.DataFrame(TPTN.T, columns=clmn, index=indx1)

        key = learning_percentage
        frames = [globals()[f'df{lpstr}'] for lpstr in learning_percentage]
        frames1 = [globals()[f'df1{lpstr}'] for lpstr in learning_percentage]
        df = pd.concat(frames, keys=key, axis=0)
        df1 = pd.concat(frames1, keys=key, axis=0)
        df.to_csv(f'pre_evaluated/Comp{db}.csv')
