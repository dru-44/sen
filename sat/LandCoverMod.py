from sat.main import *




class senM(sen):
    X_data = y_data = X_train = X_test = y_train = y_test = X_scaled = None
    dataset = 'SB'
    test_size = 0.30
    windowSize = 15
    MODEL_NAME = 'TestModel'
    K = 5
    history = None
    df_cm= None
    pred= None
    pred_t= None
    X=None 
    
    def __init__(self,Path=None):
      with console.status("[bold green]Verifying resources...") as status:
        sen.__init__(self, Path)
        
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

       
        senM.X_data = np.moveaxis(sen.arr_st, 0, -1)
        senM.y_data = loadmat('/content/'+self.Path +
                              '/Sundarbands_gt.mat')['gt']
        
        senM.X,pca = applyPCA(senM.X_data,numComponents=senM.K)
        senM.X, y = createImageCubes(senM.X, senM.y_data, windowSize=senM.windowSize)
        X_train, X_test, y_train, y_test = splitTrainTestSet(senM.X, y, testRatio = senM.test_size)
        senM.X_train = X_train.reshape(-1, senM.windowSize, senM.windowSize, senM.K, 1)
        senM.X_test = X_test.reshape(-1, senM.windowSize, senM.windowSize, senM.K, 1)
        senM.y_train = tf.keras.utils.to_categorical(y_train)
        senM.y_test = tf.keras.utils.to_categorical(y_test)
        console.log(f" Done!")
    
    def CModel(self):
      
        S = senM.windowSize
        L = senM.K
        output_units = senM.y_train.shape[1]

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
        model = tf.keras.Model(name = senM.dataset+'_Model' , inputs=input_layer, outputs=output_layer)

        senM.su=model.summary()
        
      
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
        history  = model.fit(x=senM.X_train, y=senM.y_train, 
                            batch_size=1024*6, epochs=6, 
                            validation_data=(senM.X_test, senM.y_test), callbacks = [tensorboard_callback, es, checkpoint])
        history  = pd.DataFrame(history .history )
        #predictions
        senM.pred = model.predict(senM.X_test, batch_size=1204*6, verbose=1)
        senM.pred_t = model.predict(senM.X.reshape(-1, senM.windowSize, senM.windowSize, senM.K, 1),
                              batch_size=1204*6, verbose=1)
        #plt.figure(figsize = (10,7))

        classes = [f'Class-{i}' for i in range(1, 7)]
        mat = tf.math.confusion_matrix(np.argmax(senM.y_test, 1),
                              np.argmax(senM.pred, 1))
        senM.df_cm = pd.DataFrame(mat.numpy(), index = classes, columns = classes)
          
        
          
        def Mgraph(self):
       
        

          plt.figure(figsize = (12, 6))
          plt.plot(range(len(history  ['accuracy'].values.tolist())), history  ['accuracy'].values.tolist(), label = 'Train_Accuracy')
          plt.plot(range(len(history  ['loss'].values.tolist())), history  ['loss'].values.tolist(), label = 'Train_Loss')
          plt.plot(range(len(history  ['val_accuracy'].values.tolist())), history  ['val_accuracy'].values.tolist(), label = 'Test_Accuracy')
          plt.plot(range(len(history  ['val_loss'].values.tolist())), history  ['val_loss'].values.tolist(), label = 'Test_Loss')
          plt.xlabel('Epochs')
          plt.ylabel('Value')
          plt.legend()
          plt.show()
        Mgraph(self)

    
   
    
    def Mhmap(self):
      
        sns.heatmap(senM.df_cm, annot=True, fmt='d')
        plt.title('Confusion Matrix', fontsize = 20)
        plt.show()
        
    
    def Mreport(self):
      
        print(classification_report(np.argmax(senM.y_test, 1),
                                    np.argmax(senM.pred, 1),
                                    target_names=[f'Class-{i}' for i in range(1, 7)]))
        






    def Mplot(self):
      with console.status("[bold green]plotting...") as status:
        
        
        # Visualize Groundtruth

        ep.plot_bands(np.argmax(senM.pred_t, axis=1).reshape(954, 298), 
                    cmap=ListedColormap(['darkgreen', 'green', 'black', 
                                        '#CA6F1E', 'navy', 'forestgreen']))
        plt.show()
        console.log(f" Done!")



    def __del__(self):
      pass