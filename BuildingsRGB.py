from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
import glob
import time as systime
ts_total = systime.time()


#np.set_printoptions(threshold=np.inf)


print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')







eig=73
yy=150
h=450
w=150


#np.set_printoptions(threshold=np.inf)
#The sub folders (classes) and images are loaced in "path")
path = 'seg_train'
#path = 'smol'
#Get subfolders in path. These will be classes
directory_contents = os.listdir(path)

numclasses = np.size(directory_contents)

#directory_contents = directory_contents[:2]
target_names = directory_contents

#Populate a label vector
y = []
for i in range(0,len(directory_contents)):
    newpath = path + '\\' + directory_contents[i]
    newdirectory_contents = os.listdir(newpath)
    for j in range(0,np.size(newdirectory_contents)):
        y.append(i)


#get image data
X=[]
for i in range(0,len(directory_contents)):
    newpath = os.path.join(path,directory_contents[i])


    imagelist = os.listdir(newpath)

    image_list = []
    for filename in glob.iglob(newpath + '**/*.jpg', recursive=True):
        image = Image.open(filename).convert('RGB')
        with image as im:
            # The crop method from the Image module takes four coordinates as input.
            # The right can also be represented as (left+width)
            # and lower can be represented as (upper+height).
            #(left, upper, right, lower) = (78, 62, 172, 187)
            #(left, upper, right, lower) = ((150-yy)/2, (150-yy)/2, (150-yy)/2+yy, (150-yy)/2+yy)
            (left, upper, right, lower) = (0,0,150,150)
            # Here the image "im" is cropped and assigned to new variable im_crop
            im_crop = im.crop((left, upper, right, lower))
            image = im_crop#.resize([h,w])
            arr=np.asarray(image)
            X.append(arr.flatten())

print('num images ', len(X))
print('y ', len(y))
X=np.array(X)




#Images are cropped to 94x125 pixels

n_samples=np.size(y)
n_features = X.shape[1]

# X=X.sum(axis=0)/n_samples
#
# plt.imshow(X.reshape((h, w)), cmap=plt.cm.gray)
# plt.show()

# the label to predict is the id of the person
#y = lfw_people.target
n_classes = np.size(target_names)#.shape[0]


#Cross validation step. Set seed for repeatability
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)


# #############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = eig


pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)

eigenfaces = pca.components_.reshape((n_components, h, w))


X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)



# #############################################################################
# Train a SVM classification model

ts = systime.time()
print("Fitting the classifier to the training set")
t0 = time()
#param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
param_grid = {'C': [1e3], 'gamma': [0.0001], }
clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid
)
clf = clf.fit(X_train_pca, y_train)
print(clf.best_estimator_)

te = systime.time()

# #############################################################################
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print(y_pred)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=1, n_col=1):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)),cmap=plt.cm.gray)
        plt.title(titles[i], size=8)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'Predicted: %s\nActual:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["Eigenface: %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
print(y_pred)
print(y_test)
correct = 0
total = np.size(y_test)
for i in range(0,total):
    if y_pred[i] == y_test[i]:
        correct = correct + 1

print('Accuract: ' + str(correct/total))

te_total = systime.time()
print('total time ', te_total - ts_total)
print(te - ts)

plt.show()


