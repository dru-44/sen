from main import *


class SenM(sen):
    X_data = y_data = X_train = X_test = y_train = y_test = X_scaled = None
    dataset = 'SB'
    test_size = 0.30
    windowSize = 15
    MODEL_NAME = 'TestModel'
    K = 5
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