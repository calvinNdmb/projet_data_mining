import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt

def load_data(file, header, separator):
    if header:
        data = pd.read_csv(file, sep=separator)
    else:
        data = pd.read_csv(file, sep=separator, header=None)
    return data

def nan_to_remove_columns(df):
    count_nan = df.isna().sum()
    delete_c = False
    if delete_c:
        for i in df.columns:
            if count_nan[i] != 0:
                df.drop([i], axis=1, inplace=True)
    return df

def nan_to_mean(df):
    for i in df.columns:
        if df[i].dtype == "int64" or df[i].dtype == "float64":
            df[i].fillna(df[i].mean(), inplace=True)
    return df

def nan_to_mode(df):
    for i in df.columns:
        if df[i].dtype == "object":
            df[i].fillna(df[i].mode()[0], inplace=True)
    return df

def nan_to_med(df):
    for i in df.columns:
        if df[i].dtype == "int64" or df[i].dtype == "float64":
            df[i].fillna(df[i].median(), inplace=True)
    return df

def remove_nan(df):
    df = df.dropna()
    return df

def nan_to_knn_imputer(df, numerical_columns, n_neighbors=3):
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df[numerical_columns] = knn_imputer.fit_transform(df[numerical_columns])
    return df

def min_max_normalization(df, numerical_columns):
    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

def zscore_standardization(df, numerical_columns):
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

def main():
    st.title("My Streamlit Page")
    st.write("Welcome to my Streamlit page!")

    st.sidebar.title("Sidebar")

    st.sidebar.header("Part 1")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        header = st.sidebar.checkbox("Does the file have a header?", value=True)
        separator = st.sidebar.text_input("Separator", value=",")
        data = load_data(uploaded_file, header, separator)

        st.write("Data Preview:")
        st.write(data.head())

        numerical_columns = [i for i in data.columns if data[i].dtype == "int64" or data[i].dtype == "float64"]

        # Handle missing values
        st.sidebar.subheader("Missing Values Handling")
        missing_values_option = st.sidebar.selectbox("Choose a method to handle missing values",
                                                     ["Remove Columns", "Fill with Mean", "Fill with Mode", "Fill with Median", "KNN Imputer", "Remove Rows"])
        if missing_values_option == "Remove Columns":
            data = nan_to_remove_columns(data)
        elif missing_values_option == "Fill with Mean":
            data = nan_to_mean(data)
        elif missing_values_option == "Fill with Mode":
            data = nan_to_mode(data)
        elif missing_values_option == "Fill with Median":
            data = nan_to_med(data)
        elif missing_values_option == "KNN Imputer":
            data = nan_to_knn_imputer(data, numerical_columns)
        elif missing_values_option == "Remove Rows":
            data = remove_nan(data)

        st.write("Data after handling missing values:")
        st.write(data.head())

        # Normalize or standardize data
        st.sidebar.subheader("Normalization/Standardization")
        normalization_option = st.sidebar.selectbox("Choose a method",
                                                    ["Min-Max Normalization", "Z-score Standardization", "None"])
        if normalization_option == "Min-Max Normalization":
            data = min_max_normalization(data, numerical_columns)
        elif normalization_option == "Z-score Standardization":
            data = zscore_standardization(data, numerical_columns)

        st.write("Data after normalization/standardization:")
        st.write(data.head())

        # Clustering
        st.sidebar.subheader("Clustering")
        clustering_option = st.sidebar.selectbox("Choose a clustering method", ["KMeans", "DBSCAN"])
        if clustering_option == "KMeans":
            k = st.sidebar.slider("Select number of clusters for KMeans", 1, 10, 3)
            kmeans = KMeans(n_clusters=k)
            data['Cluster'] = kmeans.fit_predict(data[numerical_columns])
        elif clustering_option == "DBSCAN":
            eps = st.sidebar.slider("Select epsilon for DBSCAN", 0.1, 1.0, 0.3)
            min_samples = st.sidebar.slider("Select min_samples for DBSCAN", 1, 10, 2)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            data['Cluster'] = dbscan.fit_predict(data[numerical_columns])

        st.write("Data after clustering:")
        st.write(data.head())

if __name__ == "__main__":
    main()
