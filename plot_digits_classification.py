"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

train_set = 0.8
dev_set = 0.1
test_set = 0.1

dev_test_frac = 1-train_set

# Split data into 50% train and 50% test subsets
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size= dev_test_frac, shuffle=True)

X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_set/dev_test_frac), shuffle=True)


#hyperparameter tuning
# gamma_list = [0.01,0.005,0.001,0.0005,0.0001]
# c_list = [0.1,0.2,0.5,0.7,1,2,5,7,10]


gamma_list = [0.001,0.05,0.00001,0.0009,0.08]
c_list = [0.2,0.5,0.01,0.9,5,3,2,0.1,0.004]

h_param_comb = [{'gamma':g,'C':c} for g in gamma_list for c in c_list]
assert len(h_param_comb) == len(gamma_list)*len(c_list)
GAMMA =0.001
C = 1.0
# Create a classifier: a support vector classifier
best_acc_dev = -1.0
best_acc_train = -1.0
best_acc_test = -1.0
best_model_dev =None
best_model_train=None
best_model_test =None
best_h_params =None
for cur_h_params in h_param_comb:
    clf = svm.SVC()
    hyper_params = cur_h_params
    clf.set_params(**hyper_params)



    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
    
    print(cur_h_params)
    #Predict the value of the digit on the dev subset
    predicted_dev = clf.predict(X_dev)
    predicted_train = clf.predict(X_train)
    predicted_test = clf.predict(X_test)

    cur_acc_dev = metrics.accuracy_score(y_pred=predicted_dev,y_true=y_dev)
    cur_acc_train = metrics.accuracy_score(y_pred=predicted_train,y_true=y_train)
    cur_acc_test = metrics.accuracy_score(y_pred=predicted_test,y_true=y_test)
    if(cur_acc_dev>best_acc_dev):
        best_acc_dev=cur_acc_dev
        best_model_dev=clf
        best_h_params=cur_h_params
        print("Found new best acc with",str(cur_h_params))
        print("New best val acc",str(cur_acc_dev))
    if(cur_acc_train>best_acc_train):
        best_acc_train=cur_acc_train
        best_model_train=clf
        best_h_params=cur_h_params
        print("Found new best acc with",str(cur_h_params))
        print("New best val acc",str(cur_acc_train))

    if(cur_acc_test>best_acc_test):
        best_acc_test=cur_acc_test
        best_model_test=clf
        best_h_params=cur_h_params
        print("Found new best acc with",str(cur_h_params))
        print("New best val acc",str(cur_acc_test))
# Predict the value of the digit on the test subset
predicted_dev = best_model_dev.predict(X_test)
predicted_train = best_model_train.predict(X_test)
predicted_test = best_model_test.predict(X_test)

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted_dev):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.
print(cur_h_params)
print(
    f"Classification report for classifier -dev{clf}:\n"
    f"{metrics.classification_report(y_test, predicted_dev)}\n"
)
print("Best hyper parameters were",cur_h_params)

print(cur_h_params)
print(
    f"Classification report for classifier -train {clf}:\n"
    f"{metrics.classification_report(y_test, predicted_train)}\n"
)
print("Best hyper parameters were",cur_h_params)

print(cur_h_params)
print(
    f"Classification report for classifier -test {clf}:\n"
    f"{metrics.classification_report(y_test, predicted_test)}\n"
)
print("Best hyper parameters were",cur_h_params)
##################################################
###############################################################################
# # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# # true digit values and the predicted digit values.

# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")

# plt.show()
