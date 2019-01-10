from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Input, Lambda, Activation, MaxPooling2D, Flatten, concatenate, BatchNormalization
from tensorflow.keras.models import Model as keras_model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from itertools import product
import glob
from tensorflow.keras.preprocessing import image
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score

#parameters:
criterias = ["gini","entropy"]
max_depths=[6,7,8,9]
min_samples_leafs = [5,10,15,20,30,50]
model = load_model('C:/Users/AYPEK/Desktop/deneme/weights-improvement-52-0.94.hdf5')
train_img_dir_metal = "C:/Users/AYPEK/Desktop/deneme/train/Metal"
train_img_dir_plastic = "C:/Users/AYPEK/Desktop/deneme/train/Plastic"
train_img_dir_bird = "C:/Users/AYPEK/Desktop/deneme/train/Bird"
test_img_dir_metal = "C:/Users/AYPEK/Desktop/deneme/validation/Metal"
test_img_dir_plastic = "C:/Users/AYPEK/Desktop/deneme/validation/Plastic"
test_img_dir_bird = "C:/Users/AYPEK/Desktop/deneme/validation/Bird"

def create_configuratins():
    configurations = [] #[criterion, max_depth, min_samples_leaf ]
    for criteria, max_depth, min_samples_leaf in product(criterias, max_depths, min_samples_leafs):
        configurations.append([criteria, max_depth, min_samples_leaf])
    return configurations

def get_features_for_given_path(path, feature_extractor_model, features, currentLabel, labelsArray):
    path+="/*.png"
    imgs = glob.glob(path)
    for img in imgs:
        curr_img = Image.open(img)
        rescaled_img = curr_img.resize((200, 200), Image.ANTIALIAS)
        grayImg = (1. / 255) * np.array(rescaled_img)
        curr_img = image.img_to_array(grayImg)
        curr_img = np.expand_dims(curr_img, axis=0)
        feature_vect = feature_extractor_model.predict(curr_img)
        features.append(feature_vect)
        labelsArray.append(currentLabel)


def get_pretrained_network_till_dense_layer(layer_name, model):
    pretrained_model = None
    for layer in model.layers:
        if layer.name == layer_name:
            output = layer.output
            pretrained_model = Model(model.input, output)
            pretrained_model.trainable = False

    return pretrained_model



feature_extractor_model = get_pretrained_network_till_dense_layer('dense_1', model)

print(feature_extractor_model.summary())

train_features = []
train_labels=[]
get_features_for_given_path(train_img_dir_bird, feature_extractor_model, train_features, 0, train_labels)
get_features_for_given_path(train_img_dir_metal, feature_extractor_model, train_features, 1, train_labels)
get_features_for_given_path(train_img_dir_plastic, feature_extractor_model, train_features, 2, train_labels)
train_features = np.asarray(train_features).reshape(len(train_features),train_features[0].shape[1])
train_labels=np.asarray(train_labels)

test_features=[]
test_labels=[]
get_features_for_given_path(test_img_dir_bird, feature_extractor_model, test_features, 0, test_labels)
get_features_for_given_path(test_img_dir_metal, feature_extractor_model, test_features, 1, test_labels)
get_features_for_given_path(test_img_dir_plastic, feature_extractor_model, test_features, 2, test_labels)
test_features = np.asarray(test_features).reshape(len(test_features),test_features[0].shape[1])
test_labels=np.asarray(test_labels)



configurations = create_configuratins()
for configuration in configurations:
    estimator = DecisionTreeClassifier(criterion=configuration[0], max_depth=configuration[1],
                                       min_samples_leaf=configuration[2])
    estimator.fit(train_features, train_labels)
    predicted_labels = estimator.predict(test_features)
    print("\n###################################### \n for configuration ", configuration)
    print("\n confusion matrix:\n", confusion_matrix(test_labels, predicted_labels))
    print("\n accuracy score: \n", accuracy_score(test_labels, predicted_labels))
    print("\n precision score: \n", np.mean(precision_score(test_labels, predicted_labels, average=None)))
    print("\n f1 score: \n", np.mean(f1_score(test_labels, predicted_labels, average=None)))
    tree.export_graphviz(estimator, out_file='_' + configuration[0]+'_'+str(configuration[1])+'_'+str(configuration[2])+'.dot')

