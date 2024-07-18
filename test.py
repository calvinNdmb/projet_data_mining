import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt
import numpy as np


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

    

    # Visualisation des données nettoyées
    st.sidebar.subheader("Visualization")
    st.write("## Visualisation")
    st.write("### Visualisation des données nettoyées")

    # Histogrammes des colonnes numériques
    numerical_data = data.select_dtypes(include=[np.number])
    if numerical_data.empty:
        st.write("No numerical columns available to plot histograms.")
    else:
        st.write("Histograms of Numerical Columns in DataFrame:")
        fig, axes = plt.subplots(figsize=(10, 6))
        numerical_data.hist(bins=10, ax=axes, grid=True)
        plt.suptitle('Histograms of Numerical Columns in DataFrame', fontsize=16)
        st.pyplot(fig)
    # Box plots des colonnes numériques
    fig, ax = plt.subplots()
    data.boxplot(column=numerical_columns, figsize=(10, 8))
    plt.suptitle('Box Plots of Numerical Columns in DataFrame', fontsize=16)
    st.pyplot(fig)

    # Méthode du coude pour déterminer le nombre optimal de clusters
    k_values = range(1, 11)
    wcss = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data[numerical_columns])
        wcss.append(kmeans.inertia_)

    fig, ax = plt.subplots()
    ax.plot(k_values, wcss, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.xticks(k_values)
    st.pyplot(fig)

    # Clustering avec KMeans
    optimal_k = 3  # Remplacer par la valeur optimale trouvée
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[numerical_columns])

    # Sélection des colonnes pour la visualisation
    st.sidebar.subheader("Select Columns for Visualization")
    available_columns = data.columns.tolist()

    selected_columns_2d = st.sidebar.multiselect("Select two columns for 2D visualization", available_columns, default=available_columns[:2])
    selected_columns_3d = st.sidebar.multiselect("Select three columns for 3D visualization", available_columns, default=available_columns[:3])

    # Visualisation 2D des clusters
    if len(selected_columns_2d) == 2:
        fig, ax = plt.subplots()
        scatter = ax.scatter(data[selected_columns_2d[0]], data[selected_columns_2d[1]], c=data['Cluster'], cmap='viridis')
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        plt.title('KMeans Clustering')
        plt.xlabel(selected_columns_2d[0])
        plt.ylabel(selected_columns_2d[1])
        st.pyplot(fig)
    else:
        st.write("Please select exactly two columns for 2D visualization.")

    # Visualisation 3D des clusters (si applicable)
    if len(selected_columns_3d) == 3:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data[selected_columns_3d[0]], data[selected_columns_3d[1]], data[selected_columns_3d[2]], 
                            c=data['Cluster'], cmap='viridis', marker='o')
        ax.set_title('KMeans Clustering')
        ax.set_xlabel(selected_columns_3d[0])
        ax.set_ylabel(selected_columns_3d[1])
        ax.set_zlabel(selected_columns_3d[2])
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster')
        st.pyplot(fig)
    else:
        st.write("Please select exactly three columns for 3D visualization.")


if __name__ == "__main__":
    main()
