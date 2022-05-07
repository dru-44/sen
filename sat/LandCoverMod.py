from main import *


class SenM(sen):
    X_data = y_data = X_train = X_test = y_train = y_test = X_scaled = None
    dataset = 'SB'
    test_size = 0.30
    windowSize = 15
    MODEL_NAME = 'TestModel'
    K = 5
    history = None
    def __init__(self,Path=None):
        sen.__init__(self, Path=None)
        
        def applyPCA(X, numComponents=75):
            newX = np.reshape(X, (-1, X.shape[2]))
            pca = PCA(n_components=numComponents, whiten=True)
            newX = pca.fit_transform(newX)
            newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
            return newX, pca

        def padWithZeros(X, margin=2):
            newX = np.zeros(
                (X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
            x_offset = margin
            y_offset = margin
            newX[x_offset:X.shape[0] + x_offset,
                 y_offset:X.shape[1] + y_offset, :] = X
            return newX

        def createImageCubes(X, y, windowSize=5, removeZeroLabels=False):
            margin = int((windowSize - 1) / 2)
            zeroPaddedX = padWithZeros(X, margin=margin)
            # split patches
            patchesData = np.zeros(
                (X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
            patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
            patchIndex = 0
            for r in range(margin, zeroPaddedX.shape[0] - margin):
                for c in range(margin, zeroPaddedX.shape[1] - margin):
                    patch = zeroPaddedX[r - margin:r +
                                        margin + 1, c - margin:c + margin + 1]
                    patchesData[patchIndex, :, :, :] = patch
                    patchesLabels[patchIndex] = y[r-margin, c-margin]
                    patchIndex = patchIndex + 1
            if removeZeroLabels:
                patchesData = patchesData[patchesLabels > 0, :, :, :]
                patchesLabels = patchesLabels[patchesLabels > 0]
                patchesLabels -= 1
            return patchesData, patchesLabels

        def splitTrainTestSet(X, y, testRatio, randomState=42):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=testRatio, random_state=randomState, stratify=y)
            return X_train, X_test, y_train, y_test

       
        SenM.X_data = np.moveaxis(sen.arr_st, 0, -1)
        SenM.y_data = loadmat('/content/'+self.Path +
                              '/Sundarbands_gt.mat')['gt']
        
        X,pca = applyPCA(SenM.X_data,numComponents=SenM.K)
        X, y = createImageCubes(X, SenM.y_data, windowSize=SenM.windowSize)
        X_train, X_test, y_train, y_test = splitTrainTestSet(X, y, testRatio = SenM.test_size)
        SenM.X_train = X_train.reshape(-1, SenM.windowSize, SenM.windowSize, SenM.K, 1)
        SenM.X_test = X_test.reshape(-1, SenM.windowSize, SenM.windowSize, SenM.K, 1)
        SenM.y_train = tf.keras.utils.to_categorical(y_train)
        SenM.y_test = tf.keras.utils.to_categorical(y_test)
    
    def CModel(self):
        S = SenM.windowSize
        L = SenM.K
        output_units = SenM.y_train.shape[1]

        ## input layer
        input_layer = tf. keras.Input((S, S, L, 1))

        ## convolutional layers
        conv_layer1 = tf.keras.layers.Conv3D(filters=16, kernel_size=(2, 2, 3), activation='relu')(input_layer)
        conv_layer2 = tf.keras.layers.Conv3D(filters=32, kernel_size=(2, 2, 3), activation='relu')(conv_layer1)
        conv2d_shape = conv_layer2.shape
        conv_layer3 = tf.keras.layers.Reshape((conv2d_shape[1], conv2d_shape[2], conv2d_shape[3]*conv2d_shape[4]))(conv_layer2)
        conv_layer4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(2,2), activation='relu')(conv_layer3)

        flatten_layer = tf.keras.layers.Flatten()(conv_layer4)

        ## fully connected layers
        dense_layer1 = tf.keras.layers.Dense(128, activation='relu')(flatten_layer)
        dense_layer1 = tf.keras.layers.Dropout(0.4)(dense_layer1)
        dense_layer2 = tf.keras.layers.Dense(64, activation='relu')(dense_layer1)
        dense_layer2 = tf.keras.layers.Dropout(0.4)(dense_layer2)
        dense_layer3 = tf.keras.layers.Dense(20, activation='relu')(dense_layer2)
        dense_layer3 = tf.keras.layers.Dropout(0.4)(dense_layer3)
        output_layer = tf.keras.layers.Dense(units=output_units, activation='softmax')(dense_layer3)
        # define the model with input layer and output layer
        model = tf.keras.Model(name = SenM.dataset+'_Model' , inputs=input_layer, outputs=output_layer)

        model.summary()

        # Compile
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])

        # Callbacks
        logdir = self.Path+"logs/" +model.name+'_'+datetime.datetime.now().strftime("%d:%m:%Y-%H:%M:%S")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',
                        min_delta = 0,
                        patience = 1,
                        verbose = 1,
                        restore_best_weights = True)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = 'Test_Model.h5', 
                                    monitor = 'val_loss', 
                                    mode ='min', 
                                    save_best_only = True,
                                    verbose = 1)
        # Fit
        SenM.history = model.fit(x=SenM.X_train, y=SenM.y_train, 
                            batch_size=1024*6, epochs=6, 
                            validation_data=(SenM.X_test, SenM.y_test), callbacks = [tensorboard_callback, es, checkpoint])

    def Mgraph(self):
        hist=SenM.history
        histdf= pd.DataFrame(hist.hist)

        plt.figure(figsize = (12, 6))
        plt.plot(range(len(histdf['accuracy'].values.tolist())), histdf['accuracy'].values.tolist(), label = 'Train_Accuracy')
        plt.plot(range(len(histdf['loss'].values.tolist())), histdf['loss'].values.tolist(), label = 'Train_Loss')
        plt.plot(range(len(histdf['val_accuracy'].values.tolist())), histdf['val_accuracy'].values.tolist(), label = 'Test_Accuracy')
        plt.plot(range(len(histdf['val_loss'].values.tolist())), histdf['val_loss'].values.tolist(), label = 'Test_Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        


