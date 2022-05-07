from main import *


class SenC(sen):
    X_data = y_data = X_train = X_test = y_train = y_test = X_scaled = None

    def __init__(self,Path=None):
        sen.__init__(self, Path=None)
        x = np.moveaxis(sen.arr_st, 0, -1)
        SenC.X_data = x.reshape(-1, 12)

        scaler = StandardScaler().fit(SenC.X_data)

        SenC.X_scaled = scaler.transform(SenC.X_data)
        SenC.y_data = loadmat('/content/'+self.Path +
                              '/Sundarbands_gt.mat')['gt']
        SenC.X_train, SenC.X_test, SenC.y_train, SenC.y_test = train_test_split(
            SenC.X_scaled, SenC.y_data.ravel(), test_size=0.30, stratify=SenC.y_data.ravel())

    def KNNClassifier(self):
        knn = KNeighborsClassifier(n_neighbors=6)
        knn.fit(SenC.X_train, SenC.y_train)
        knn_pred = knn.predict(SenC.X_test)
        print(f"Accuracy: {accuracy_score(SenC.y_test, knn_pred)*100}")
        print(classification_report(SenC.y_test, knn_pred))
        ep.plot_bands(knn.predict(SenC.X_scaled).reshape((954, 298)),
                      cmap=ListedColormap(['darkgreen', 'green', 'black',
                                           '#CA6F1E', 'navy', 'forestgreen']))
        plt.show()

    def SVM(self):
        svm = SVC(C=3.0, kernel='rbf', degree=6, cache_size=1024)
        svm_pred = svm.predict(SenC.X_test)
        print(f"Accuracy: {accuracy_score(SenC.y_test, svm_pred)*100}")
        print(classification_report(SenC.y_test, svm_pred))
        ep.plot_bands(svm.predict(SenC.X_scaled).reshape((954, 298)),
                      cmap=ListedColormap(['darkgreen', 'green', 'black',
                                           '#CA6F1E', 'navy', 'forestgreen']))
        plt.show()

    def LBGM(self):
        d_train = lgb.Dataset(SenC.X_train, label=SenC.y_train)

        params = {}
        params['learning_rate'] = 0.03
        params['boosting_type'] = 'gbdt'  # GradientBoostingDecisionTree
        params['objective'] = 'multiclass'  # Multi-class target feature
        params['metric'] = 'multi_logloss'  # metric for multi-class
        params['max_depth'] = 15
        # no.of unique values in the target class not inclusive of the end value
        params['num_class'] = 6

        clf = lgb.train(params, d_train, 100)
        lgb_predictions = clf.predict(SenC.X_test)
        lgb_pred = np.argmax(lgb_predictions, axis=1)
        print(f"Accuracy: {accuracy_score(SenC.y_test, lgb_pred)*100}")
        print(classification_report(SenC.y_test, lgb_pred))
        ep.plot_bands(np.argmax(clf.predict(SenC.X_scaled), axis=1).reshape((954, 298)),
                      cmap=ListedColormap(['darkgreen', 'green', 'black',
                                           '#CA6F1E', 'navy', 'forestgreen']))
        plt.show()

