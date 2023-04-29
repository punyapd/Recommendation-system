import streamlit as st
import os
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import math

from keras import models
from tensorflow.keras.preprocessing import image


model = models.load_model('finalcnnmodel.h5')
print(model.summary())

st.title("Wearable Items Suggestion system")

DATASET_PATH = "dataset/"


df = pd.read_csv(DATASET_PATH + "selected_styles.csv",
                 nrows=24000, error_bad_lines=False)
df.head(10)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df.head(10)


# building another dataframe with filename and image type
augmentedDataframe = pd.DataFrame({
    'filename': df['image'],
    'type': df['articleType']
})

# total number of entries in the dataframe
total_row = len(augmentedDataframe)
print('total row count: ', total_row)

augmentedDataframe.head(1000)

# creating the list with unique values
unique_types = augmentedDataframe['type'].unique().tolist()


print(unique_types)


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


# predict the class of input image
def make_prediction(img_path):
    img = image.load_img(img_path)
    img_array = image.img_to_array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(img_array, dsize=(28, 28))
    x_data = np.array(resized_img).reshape(-1, 28, 28, 1)
    x_data = x_data/255
    print(x_data)
#     print(x_data.shape)
    result = model.predict(x_data)
    print("===========================================")
    print(np.argmax(result))
    print(unique_types[1])
    return x_data, unique_types[np.argmax(result)]

# function for calculating distance


def calculateDistance(i1, i2):
    return math.sqrt(np.sum((i1-i2)**2))


# upload a file
uploaded_file = st.file_uploader("Choose an Image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        display_image = Image.open(uploaded_file)
        # image = cv2.imread(uploaded_file, cv2.IMREAD_COLOR)
        # resized_img = cv2.resize(uploaded_file, dsize=(224, 224))
        st.image(uploaded_file)
        numpy_image, result = make_prediction(uploaded_file)

        # checking for similarrity
        typeList = []
        for i, row in df.iterrows():
            if(row["articleType"] == result):
                print(row["id"], row["articleType"])

                typeList.append(row['id'])
                print("typelsit++++++++++++++++++++++++++++++++++++++")
                print(typeList, "dfjsdfjsdj")

            i = 0
            X_similar = []
            X_id_similar = []
            X_numpy = []
        for imageId in typeList:
            Image_path = DATASET_PATH+"selected"+"/"+str(imageId)+".jpg"
            image = cv2.imread(Image_path, cv2.IMREAD_GRAYSCALE)
            try:
                resized_img = cv2.resize(image, dsize=(28, 28))
            except:
                print("can't read file: ", str(imageId)+".jpg")
            X_similar.append(resized_img)
            X_id_similar.append(imageId)

        X_numpy = np.array(X_similar).reshape(-1, 28, 28, 1)
        X_numpy = X_numpy/255

        print(calculateDistance(numpy_image, X_numpy[0]))

        # finding 10 simillar items
        distance_list = []
        for i in range(0, len(X_numpy)):
            distance_list.append(calculateDistance(numpy_image, X_numpy[i]))

        sorted_distance_list = distance_list.copy()
        # print(distance_list)
        sorted_distance_list.sort()

        least_ten_distance = sorted_distance_list[0:10]
        print(least_ten_distance)
        index_distance = []
        for i in range(0, len(least_ten_distance)-1):
            if(least_ten_distance[i] != least_ten_distance[i+1]):
                index_distance.append(
                    distance_list.index(least_ten_distance[i]))

        index_distance = index_distance[0:5]

        print(index_distance)

        # show
        col1, col2, col3, col4, col5 = st.columns(5)
        Image_path = DATASET_PATH+"selected"+"/"

        with col1:
            st.image(Image_path + str(X_id_similar[index_distance[0]])+".jpg")
        with col2:
            st.image(Image_path + str(X_id_similar[index_distance[1]])+".jpg")

        with col3:
            st.image(Image_path + str(X_id_similar[index_distance[2]])+".jpg")

        with col4:
            st.image(Image_path + str(X_id_similar[index_distance[3]])+".jpg")

        with col5:
            st.image(Image_path + str(X_id_similar[index_distance[4]])+".jpg")

    else:
        st.header("some error occured")
