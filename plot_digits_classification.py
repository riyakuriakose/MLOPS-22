"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from utils import preprocess_digits,data_viz,train_dev_test_split,h_param_tuning,pred_image_viz

gamma_list = [0.001,0.05,0.00001,0.0009,0.08]
c_list = [0.2,0.5,0.01,0.9,5,3,2,0.1,0.004]

h_param_comb = [{'gamma':g,'C':c} for g in gamma_list for c in c_list]
assert len(h_param_comb) == len(gamma_list)*len(c_list)

digits = datasets.load_digits()
data_viz(digits)

data,label = preprocess_digits(digits)

train_frac,dev_frac,test_frac = 0.8,0.1,0.1
assert train_frac+dev_frac+test_frac==1.0

x_train,y_train,x_dev,y_dev,x_test,y_test = train_dev_test_split(data,label,train_frac,dev_frac)



clf = svm.SVC()
metric = metrics.accuracy_score

best_model,best_acc,best_h_params  = h_param_tuning(h_param_comb,clf,x_train,y_train,x_dev,y_dev,metric)



predicted = best_model.predict(x_test)
pred_image_viz(predicted,x_test)
    


print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()

print("Best hyperparameters were: ")
print(best_h_params)



