


import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
def preprocess_digits(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data,label

def data_viz(dataset):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, dataset.images, dataset.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)


def train_dev_test_split(data,label,train_frac,dev_frac):
    dev_test_frac = 1-train_frac

    # Split data into 50% train and 50% test subsets
    x_train, x_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size= dev_test_frac, shuffle=True)

    x_test, x_dev, y_test, y_dev = train_test_split(
        x_dev_test, y_dev_test, test_size=(dev_frac/dev_test_frac), shuffle=True)
    return  x_train,y_train,x_dev,y_dev,x_test,y_test


def h_param_tuning(h_params_comb,clf,x_train,y_train,x_dev,y_dev,metric):
        best_metric = -1.0
        best_model = None
        best_h_params = None
        for cur_h_params in h_params_comb:
            GAMMA = 0.001
            C = 0.5
           

            #part: setting up hyperparameter
            hyper_params = {'gamma':GAMMA,"C":C}
            clf.set_params(**hyper_params)




            # Learn the digits on the train subset
            clf.fit(x_train, y_train)
        
            #PART: 
            # Predict the value of the digit on the test subset
            predicted_dev = clf.predict(x_dev)
            cur_metric = metric(y_pred = predicted_dev,y_true = y_dev)

            if cur_metric > best_metric:
                best_acc = cur_metric
                best_model = clf
                best_h_params = cur_h_params
                print("found new best acc with: "+str(cur_h_params))
                print("New best val accuracy:"+str(cur_metric))
        return best_model,best_acc,best_h_params


def pred_image_viz(predictions,x_test):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, x_test, predictions):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

