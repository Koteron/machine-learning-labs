import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, auc, f1_score, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics


x1_negative = np.random.normal(20, 3, 10)
x2_negative = np.random.normal(4, 3, 10)
x1_positive = np.random.normal(11, 1, 90)
x2_positive = np.random.normal(18, 1, 90)
ax = plt.subplots()
plt.scatter(x1_negative, x2_negative, color='red')
plt.scatter(x1_positive, x2_positive, color='blue')
plt.show()

X = list()
y = list()

for i in range(0, 10):
    X.append([x1_negative[i], x2_negative[i]])
    y.append(-1)

for i in range(0, 90):
    X.append([x1_positive[i], x2_positive[i]])
    y.append(1)

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8)
GaussianNB_object = GaussianNB()
GaussianNB_object.fit(train_X, train_y)
y_prediction_test = GaussianNB_object.predict(test_X)
y_prediction_proba_test = GaussianNB_object.predict_proba(test_X)
accuracy = accuracy_score(test_y, y_prediction_test)
confusion_matrix = confusion_matrix(test_y, y_prediction_test)
print(accuracy)
print(confusion_matrix)
fpr, tpr, _ = metrics.roc_curve(test_y, y_prediction_proba_test[:, 1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange',
label='ROC кривая (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='-', label='Random Gauss')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.show()


precision, recall, _ = metrics.precision_recall_curve(test_y, y_prediction_proba_test[:, 1])
f1 = f1_score(test_y, y_prediction_test)
auc = auc(recall, precision)
ap = average_precision_score(test_y, y_prediction_proba_test[:, 1])
print('F1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
plt.plot([0, 1], [0.5, 0.5], linestyle='--', color='navy', label='Random choice')
plt.plot(recall, precision, color='darkorange', marker='.', label='PR-curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR-кривая')
plt.legend()
plt.show()