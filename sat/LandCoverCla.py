from main import *





class senC(sen):
    X_data = y_data = X_train = X_test = y_train = y_test = X_scaled = None

    def __init__(self,Path=None):
        sen.__init__(self, Path)
        x = np.moveaxis(sen.arr_st, 0, -1)
        senC.X_data = x.reshape(-1, 12)

        scaler = StandardScaler().fit(senC.X_data)

        senC.X_scaled = scaler.transform(senC.X_data)
        senC.y_data = loadmat(self.Path +
                              "/Sundarbands_gt.mat")['gt']
        senC.X_train, senC.X_test, senC.y_train, senC.y_test = train_test_split(
            senC.X_scaled, senC.y_data.ravel(), test_size=0.30, stratify=senC.y_data.ravel())

    def KNNClassifier(self):
        with console.status("[bold green]Working on task...") as status:
          knn = KNeighborsClassifier(n_neighbors=6)
          knn.fit(senC.X_train, senC.y_train)
          knn_pred = knn.predict(senC.X_test)
          
          print(f"Accuracy: {accuracy_score(senC.y_test, knn_pred)*100}\n")
          print(classification_report(senC.y_test, knn_pred))
        with console.status("[bold green]plotting...") as status:
                ep.plot_bands(knn.predict(senC.X_scaled).reshape((954, 298)),
                              cmap=ListedColormap(['darkgreen', 'green', 'black',
                                                  '#CA6F1E', 'navy', 'forestgreen']),title="K-NNC")
                plt.show()
                console.log(f" Done!")
        

    def SVM(self):
      with console.status("[bold green]Working on task...") as status:
        svm = SVC(C=3.0, kernel='rbf', degree=6, cache_size=1024)
        svm.fit(senC.X_train, senC.y_train)
        svm_pred = svm.predict(senC.X_test)
        print(f"Accuracy: {accuracy_score(senC.y_test, svm_pred)*100}")
        print(classification_report(senC.y_test, svm_pred))
      with console.status("[bold green]plotting...") as status:
        ep.plot_bands(svm.predict(senC.X_scaled).reshape((954, 298)),
                      cmap=ListedColormap(['darkgreen', 'green', 'black',
                                           '#CA6F1E', 'navy', 'forestgreen']),title="SVM")
        plt.show()
        console.log(f" Done!")

    def LBGM(self):
      with console.status("[bold green]Working on task...") as status:
        d_train = lgb.Dataset(senC.X_train, label=senC.y_train)

        params = {}
        params['learning_rate'] = 0.03
        params['boosting_type'] = 'gbdt'  # GradientBoostingDecisionTree
        params['objective'] = 'multiclass'  # Multi-class target feature
        params['metric'] = 'multi_logloss'  # metric for multi-class
        params['max_depth'] = 15
        # no.of unique values in the target class not inclusive of the end value
        params['num_class'] = 6

        clf = lgb.train(params, d_train, 100)
        lgb_predictions = clf.predict(senC.X_test)
        lgb_pred = np.argmax(lgb_predictions, axis=1)
        print(f"Accuracy: {accuracy_score(senC.y_test, lgb_pred)*100}")
        print(classification_report(senC.y_test, lgb_pred))
      with console.status("[bold green]plotting...") as status:
        print("\n\n")
        ep.plot_bands(np.argmax(clf.predict(senC.X_scaled), axis=1).reshape((954, 298)),
                      cmap=ListedColormap(['darkgreen', 'green', 'black',
                                           '#CA6F1E', 'navy', 'forestgreen']),title="LBGM")
        plt.show()
        console.log(f" Done!")

