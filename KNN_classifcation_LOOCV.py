import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import hog
import pandas as pd
import seaborn as sn

image_path = "training"
class_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
IMG_SIZE = 28  # number of pixels "rows=columns"
training_data = []  # a list used to save all training images along with their labels
test_images_number = 200  # total number of test images

def create_training_data():  # imports images from each digit's folder
    for class_name in class_names:
        path = os.path.join(image_path, class_name)  # create path each digit
        class_num = class_names.index(class_name)  # labelling
        print(class_num)
        for img in tqdm(os.listdir(path)):  # iterate over each image in each digit
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # reads images and to makes sure that all images are processed in the same way "no colored images"
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                filtered_array = cv2.medianBlur(resized_array, 3)  # using a median filter to remove salt and pepper noise
                hog_features, hog_image = hog(filtered_array, visualize=True, block_norm='L2-Hys', pixels_per_cell=(6, 6))
                training_data.append([hog_features, class_num])  # add this to our training_data
            except Exception as e:  # for the interest in keeping the output clean...
                pass
    original_features_list = []
    original_labels_list = []
    for features, label in training_data:
        original_features_list.append(features)
        original_labels_list.append(label)
    original_features = np.array(original_features_list, dtype=np.float64)
    original_labels = np.array(original_labels_list)
    return original_features, original_labels

def LOOCV(features_original, labels_original):  # Leave One Out Cross Validation
    k_max = 100  # max number of K's trials
    k_count = np.zeros([features_original.shape[0], k_max], dtype=np.float64)  # number of rows= total number of validation samples, number of columns = total number of k "100"
    distances_saved = saved_distances(features_original)  # getting the saved distances
    k_values = np.arange(start=1, stop=101, step=1)  # creating an array containing the choice of k
    for i in range(features_original.shape[0]):  # 2400 samples
        labels_temp_validation = labels_original[i]  # temporary array containing the label for the validation sample
        labels_temp_testing = np.delete(labels_original, i, 0)  # array containing all the labels minus the label of the validation sample
        print("validation sample: " + str(i))
        for j in range(0, k_max):  # 100 k
            class_predicted = predict_KNN(distances_saved[:, i], labels_temp_testing, j + 1)  # classification result
            if class_predicted == labels_temp_validation:  # recording if it was classified correctly or not
                k_count[i, j] = 1
            else:
                k_count[i, j] = 0
    k_count_average = np.mean(k_count, axis=0)  # array containing accuracy of each value of K
    k_count_error = 1 - k_count_average  # array containing classification error of each value of K
    plt.title('Classification error vs choice of K ')
    plt.scatter(k_values, k_count_error, color='red')
    plt.plot(k_values, k_count_error, color='red')
    plt.xlabel('choice of K')
    plt.ylabel('Classification error')
    plt.grid()
    plt.show()
    for i in range(k_max):
        print("K value:" + str(i + 1) + " has a classification error of:" + str(k_count_error[i]))
    best_k = np.argmin(k_count_error) + 1  # getting the value of k with the lease classification error
    plt.bar(k_values, k_count_error, color='green')
    plt.xlabel("Choice of K")
    plt.ylabel("Classification error")
    plt.title("Classification error vs Choice of K")

    plt.show()
    return best_k

def Testing_function(features_original, labels_original, test_images_number):
    elements = np.arange(0, 10, 1)
    labels_testing = np.repeat(elements, 20)  # creating an array containing labels in order from 0 to 9 "ex: 000000...00 111111111...11"
    confusion_matrix = np.zeros([10, 10], dtype=np.uint8)
    for i in range(test_images_number):  # total number of test images
        image_name = "N" + str(i + 1) + ".jpg"
        path = "Test\\" + image_name
        image_original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # to make sure that all image are processed in the same way "no colored images"
        resized_array = cv2.resize(image_original, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)  # resize to normalize data size in case that images are not 28x28
        filtered_array = cv2.medianBlur(resized_array, 3)  # using a median filter to remove salt and pepper noise
        hog_feature1, hog_image1 = hog(filtered_array, visualize=True, block_norm='L2-Hys', pixels_per_cell=(6, 6))  # extraxting HOG features
        rows, columns = features_original.shape
        distance = np.zeros([rows, columns], dtype=np.float64)
        for j in range(0, features_original.shape[0]):
            distance[j, :] = abs(hog_feature1 - features_original[j, :])
        distance = np.sum(distance, axis=1)
        predicted_class = predict_KNN(distance, labels_original, 4, image_original, 1, image_name, "Results")
        confusion_matrix[labels_testing[i], predicted_class] = confusion_matrix[labels_testing[i], predicted_class] + 1  # filling the confusion matrix
    df_cm = pd.DataFrame(confusion_matrix, range(10), range(10))
    sn.set(font_scale=1.4)  # for label size
    ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size
    ax.set_title('Confusion matrix')
    ax.set_xlabel(" Actual")
    ax.set_ylabel("predicted")
    plt.show()  # plotting confusion matrix
    return confusion_matrix

def saved_distances(features_original):  # saving the distanes as advised in the PDF
    distance_saved = np.zeros([features_original.shape[0] - 1, features_original.shape[0]], dtype=np.float64)  # an array containing all the distances
    for i in range(features_original.shape[0]):
        features_temp_validation = features_original[i, :]  # an array containing the feature vector of the validation sample
        features_temp_testing = np.delete(features_original, i, 0)  # an array containing all of the feature vectors of the tesing samples minus the feature vector of the validation sample
        distance = np.zeros([features_temp_testing.shape[0], features_original.shape[1]], dtype=np.float64)  # an array containing the difference in distance between the feature vector of the validation sample and the feature vector of one  of the testing samples
        for j in range(features_temp_testing.shape[0]):
            distance[j, :] = abs(features_temp_validation - features_temp_testing[j, :])
        distance = np.sum(distance, axis=1)
        distance_saved[:, i] = distance
    return distance_saved

def predict_KNN(distance, labels, k, image=np.zeros([28, 28]), to_be_saved=0, saving_name="default.jpg", path_to_be_saved="Results"):  # KNN classifier, the function saves the classification results as text over each image "if needed"
    new_order = np.lexsort([labels, distance])  # sorting the distances to know class that the image belongs to according to 'k" neighbors
    distance = distance[new_order]
    labels = labels[new_order]
    distance = distance[0:k]
    labels = labels[0:k]
    predictedclass = np.bincount(labels).argmax()
    name = class_names[predictedclass]
    if to_be_saved == 1:
        image_final = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
        cv2.putText(image_final, name, (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(path_to_be_saved, saving_name), image_final)
    return predictedclass  # returning the classification result

original_features, original_labels = create_training_data() # get the features and corrosponding labels for the training dataset
confusion_matrix = np.zeros([10, 10], dtype=np.uint8)
k=LOOCV(original_features, original_labels)
confusion_matrix = Testing_function(original_features, original_labels, test_images_number)
print(confusion_matrix)
