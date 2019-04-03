import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

training = pd.DataFrame(pd.read_csv('/Users/Monisha/Desktop/training_v2.csv'))
test = pd.DataFrame(pd.read_csv('/Users/Monisha/Desktop/test.csv'))

training_data = training.iloc[:, 2:21]
training_target = training.iloc[:, 1]
test_data = test.iloc[:, 2:21]
test_target = test.iloc[:, 1]

parameters = {'hidden_layer_sizes': [30, 50, 70],
              'activation': ['identity', 'logistic', 'relu'],
              'max_iter': [100, 200, 500],
              #'alpha': [0.0001, 0.0005],
              'learning_rate': ['constant', 'invscaling', 'adaptive']
              }

mlp = MLPClassifier()
clf = GridSearchCV(mlp, parameters, cv=5)
clf.fit(training_data, training_target)

output = pd.DataFrame(clf.cv_results_)
output = output[output.columns[3:11]]
output.to_csv('/Users/Diana/Desktop/NN_Output.csv', sep=',')

sorted(clf.cv_results_.keys())

print('1. Best Parameters on Training Set: \n', clf.best_params_, '\n')
print('2. Best Estimator on Training Set: \n'+ clf.best_estimator_, '\n')
print('3. Best Score on Training Set: \n', clf.best_score_, '\n')

predictions = clf.predict(test_data)
print('4. Confusion Matrix on Validation Set: \n', metrics.confusion_matrix(test_target, predictions), '\n')
print('5. Accuracy on Validation Set: \n', np.mean(predictions == test_target), '\n')
print('6. Classification Report on Validation Set: \n', metrics.classification_report(test_target, predictions), '\n')

# auc, roc
fpr, tpr, thresholds = roc_curve(test_target, predictions)
roc_auc = auc(fpr, tpr)

plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()