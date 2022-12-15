import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
path = 'Breast_cancer1.h5'
data= tf.keras.models.load_model(path )
scalar=StandardScaler()
from sklearn import datasets
dataset=datasets.load_breast_cancer()
data_df = pd.DataFrame(data = dataset.data, columns = dataset.feature_names)
data_df['target']=pd.DataFrame(data=dataset.target)
x=data_df.drop(['target'],axis=1)
y=data_df['target']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
x_train=scalar.fit_transform(x_train)

def Prediction(input_data):
    data_asarray = np.asarray(input_data)
    data_reshape = data_asarray.reshape(1, -1)
    input_data_std = scalar.transform(data_reshape)
    Prediction = data.predict(input_data_std)
    Prediction_label = np.argmax(Prediction)
    if (Prediction_label == 0):
        return 'The tumor is Malignant'

    else:
         return 'The tumor is Benign'
def main():
    st.title("Breast Cancer Prediction")
    mean_radius = st.text_input("Mean Radius")
    mean_texture = st.text_input("Mean Texture")
    mean_perimeter = st.text_input("Mean Perimeter")
    mean_area = st.text_input("Mean Area")
    mean_smoothness = st.text_input("Mean Smoothness")
    mean_compactness = st.text_input("Mean Compactness")
    mean_concavity = st.text_input("Mean Concavity")
    mean_concave_points = st.text_input("Mean Concave Points")
    mean_symmetry = st.text_input("Mean Symmetry")
    mean_fractal_dimension = st.text_input("Mean Fractal Dimensions")
    radius_error = st.text_input("radius_error")
    texture_error = st.text_input("texture_erro")
    perimeter_error = st.text_input("perimeter_error")
    area_error = st.text_input("area_error  ")
    smoothness_error = st.text_input("smoothness_error ")
    compactness_error = st.text_input("compactness_error ")
    concavity_error = st.text_input("concavity_error  ")
    concave_points_error = st.text_input("concave_points_error ")
    symmetry_error = st.text_input("symmetry_error")
    fractal_dimension_error = st.text_input("fractal_dimension_error")
    worst_radius = st.text_input("Worst Radius")
    worst_texture = st.text_input("WorstTexture")
    worst_perimeter = st.text_input("worst Perimeter")
    worst_area = st.text_input("worst Area")
    worst_smoothness = st.text_input("worst Smoothness")
    worst_compactness = st.text_input("worst Compactness")
    worst_concavity = st.text_input("worstConcavity")
    worst_concave_points = st.text_input("worst Concave Points")
    worst_symmetry = st.text_input("worst Symmetry")
    worst_fractal_dimension = st.text_input("worst Fractal Dimensions")

    Predict = ''
    if st.button("Prediction Result"):
        input_data = (
        mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity,
        mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error,
        area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error,
        fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness,
        worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension)
        Predict = Prediction(input_data)
    st.success(Predict)


if __name__=='__main__':
    main()
