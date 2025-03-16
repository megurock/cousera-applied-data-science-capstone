import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def plot_confusion_matrix(y,y_predict):
  "this function plots the confusion matrix"
  from sklearn.metrics import confusion_matrix

  cm = confusion_matrix(y, y_predict)
  ax= plt.subplot()
  sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
  ax.set_xlabel('Predicted labels')
  ax.set_ylabel('True labels')
  ax.set_title('Confusion Matrix')
  ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed'])
  plt.show()


data = pd.read_csv("./dataset_part_2.csv")
print(data.head())
print(data.shape)

X = pd.read_csv("./dataset_part_3.csv")
print(X.head())

# --------------------------------------------------------------------------------
# TASK 1: Create a NumPy array from the column Class in data, by applying the method to_numpy() then assign it to the variable Y,make sure the output is a Pandas series (only one bracket df['name of column']).
# --------------------------------------------------------------------------------
Y = data["Class"].to_numpy()
print(Y[:5])  # 最初の5個を表示
print(type(Y))  # データ型を確認



# --------------------------------------------------------------------------------
# TASK 2: Standardize the data in X then reassign it to the variable X using the transform provided below.
# students get this
# transform = preprocessing.StandardScaler()
# --------------------------------------------------------------------------------
# X のカラム名を保存
column_names = X.columns

# StandardScaler を適用
transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)  # NumPy 配列が返されます

# NumPy 配列から DataFrame に戻す
X = pd.DataFrame(X, columns=column_names)

# 確認
print(X.head())


# --------------------------------------------------------------------------------
# TASK 3: Use the function train_test_split to split the data X and Y into training and test data. Set the parameter test_size to 0.2 and random_state to 2. The training data and test data should be assigned to the following labels.
# X_train, X_test, Y_train, Y_test
# --------------------------------------------------------------------------------
# データを訓練用データとテスト用データに分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# 分割したデータの形状を確認
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)



# --------------------------------------------------------------------------------
# TASK 4: Create a logistic regression object then create a GridSearchCV object logreg_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.
# --------------------------------------------------------------------------------
# ロジスティック回帰のインスタンスを作成
lr = LogisticRegression()

# ハイパーパラメータの辞書
parameters = {'C': [0.01, 0.1, 1],
              'penalty': ['l2'],
              'solver': ['lbfgs']}  # l1 lasso l2 ridge

# GridSearchCVを作成（交差検証は10-fold）
logreg_cv = GridSearchCV(estimator=lr, param_grid=parameters, cv=10)

# モデルを訓練データにフィット
logreg_cv.fit(X_train, Y_train)

# 最適なハイパーパラメータと検証データでの精度を表示
print("tuned hyperparameters (best parameters):", logreg_cv.best_params_)
print("accuracy:", logreg_cv.best_score_)


# --------------------------------------------------------------------------------
# TASK 5: Calculate the accuracy on the test data using the method score:
# --------------------------------------------------------------------------------
# テストデータに対する予測
yhat = logreg_cv.predict(X_test)

# 精度を計算
accuracy = logreg_cv.score(X_test, Y_test)
print("Accuracy on test data:", accuracy)

# 混同行列をプロット
plot_confusion_matrix(Y_test, yhat)




# --------------------------------------------------------------------------------
# TASK 6: Create a support vector machine object then create a GridSearchCV object svm_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.
# --------------------------------------------------------------------------------
# SVMのパラメータ設定
parameters = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma': np.logspace(-3, 3, 5)}

# SVMオブジェクトを作成
svm = SVC()

# GridSearchCVオブジェクトを作成（交差検証の分割数は10）
svm_cv = GridSearchCV(svm, parameters, cv=10)

# SVMモデルに対してGridSearchCVをフィット
svm_cv.fit(X_train, Y_train)

# 最適なパラメータとスコアを表示
print("tuned hyperparameters :(best parameters) ", svm_cv.best_params_)
print("accuracy :", svm_cv.best_score_)


# --------------------------------------------------------------------------------
# TASK 7: Calculate the accuracy on the test data using the method score:
# --------------------------------------------------------------------------------

# テストデータで予測を行う
yhat = svm_cv.predict(X_test)

# 精度を計算
accuracy = svm_cv.score(X_test, Y_test)
print("Accuracy on test data:", accuracy)

# 混同行列をプロット
plot_confusion_matrix(Y_test, yhat)




# --------------------------------------------------------------------------------
# TASK 8: Create a decision tree classifier object then create a GridSearchCV object tree_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.
# --------------------------------------------------------------------------------
parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()

# GridSearchCVオブジェクトを作成
tree_cv = GridSearchCV(estimator=tree, param_grid=parameters, cv=10)

# 訓練データでモデルをフィット
tree_cv.fit(X_train, Y_train)

print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)



# --------------------------------------------------------------------------------
# TASK 9: Calculate the accuracy of tree_cv on the test data using the method score:
# --------------------------------------------------------------------------------
# テストデータに対する精度を計算
accuracy = tree_cv.score(X_test, Y_test)
print("Accuracy on test data:", accuracy)

# 予測結果を取得
yhat = tree_cv.predict(X_test)

# 混同行列をプロット
plot_confusion_matrix(Y_test, yhat)




# --------------------------------------------------------------------------------
# TASK 10: Create a k nearest neighbors object then create a GridSearchCV object knn_cv with cv = 10. Fit the object to find the best parameters from the dictionary parameters.
# --------------------------------------------------------------------------------

parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier()

# GridSearchCVオブジェクトの作成（cv=10でクロスバリデーション）
knn_cv = GridSearchCV(KNN, parameters, cv=10)

# グリッドサーチをデータに適用して学習
knn_cv.fit(X_train, Y_train)
print("tuned hpyerparameters :(best parameters) ",knn_cv.best_params_)
print("accuracy :",knn_cv.best_score_)



# --------------------------------------------------------------------------------
# TASK 11: Calculate the accuracy of knn_cv on the test data using the method score:
# --------------------------------------------------------------------------------

# テストデータに対して予測を行う
yhat = knn_cv.predict(X_test)

# 精度を計算する
accuracy = knn_cv.score(X_test, Y_test)
print("Test accuracy of knn_cv:", accuracy)

# 混同行列をプロットする
plot_confusion_matrix(Y_test, yhat)


# --------------------------------------------------------------------------------
# TASK 12: Find the method performs best:
# --------------------------------------------------------------------------------
# 各モデルのテストデータに対する精度を計算する

# ロジスティック回帰モデル
logreg_accuracy = logreg_cv.score(X_test, Y_test)

# サポートベクターマシンモデル
svm_accuracy = svm_cv.score(X_test, Y_test)

# 決定木モデル
tree_accuracy = tree_cv.score(X_test, Y_test)

# K近傍法モデル
knn_accuracy = knn_cv.score(X_test, Y_test)

# 各モデルの精度を表示
print("Logistic Regression Test Accuracy:", logreg_accuracy)
print("SVM Test Accuracy:", svm_accuracy)
print("Decision Tree Test Accuracy:", tree_accuracy)
print("KNN Test Accuracy:", knn_accuracy)

# 最も良いモデルを特定
best_model = max(
    [("Logistic Regression", logreg_accuracy),
     ("SVM", svm_accuracy),
     ("Decision Tree", tree_accuracy),
     ("KNN", knn_accuracy)],
    key=lambda x: x[1]
)

print(f"The best performing model is: {best_model[0]} with an accuracy of {best_model[1]}")
