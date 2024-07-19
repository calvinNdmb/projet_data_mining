# Importation of the useful libraries
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import DBSCAN


# Data loading : Users must be able to load their own data set in CSV format. The header and the type of separation should be added by the user.
def load_data(file, header, separator):
    if header:
        data = pd.read_csv(file, sep=separator)
    else:
        data = pd.read_csv(file, sep=separator, header=None)
    return data

# Data description : Display a preview of the first and last lines of data to check their correct loading.
def display_data(data):
    st.write("First 5 rows:")
    st.write(data.head())
    st.write("Last 5 rows:")
    st.write(data.tail())

# Statistical summary : Provide a basic statistical summary of the data, including the number of lines and columns, the name of the columns, the number of missing values per column, etc.
def data_summary(data):
    st.write("Data Summary:")
    st.write(data.describe())
    st.write("Number of lines and columns:")
    st.write(f"Lines: {data.shape[0]}, Columns: {data.shape[1]}")
    st.write("Column names:")
    st.write(data.columns)
    st.write("Missing values per column:")
    st.write(data.isnull().sum())

# Suppression des colonnes avec des valeurs manquantes
def delete_columns_with_nan(data):
    data_copy = data.copy()
    count_nan = data_copy.isna().sum()
    for col in data_copy.columns:
        if count_nan[col] != 0:
            data_copy.drop([col], axis=1, inplace=True)
    return data_copy

# Suppression des lignes avec des valeurs manquantes
def delete_rows_with_nan(data):
    return data.dropna().copy()

# Remplacer les NaN par la moyenne
def nan_to_mean(data):
    imputer = SimpleImputer(strategy='mean')
    return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Remplacer les NaN par la médiane
def nan_to_median(data):
    imputer = SimpleImputer(strategy='median')
    return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Remplacer les NaN par le mode
def nan_to_mode(data):
    imputer = SimpleImputer(strategy='most_frequent')
    return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Remplacer les NaN par l'imputation KNN
def nan_to_knn_imputer(data, n_neighbors=3):
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    return pd.DataFrame(knn_imputer.fit_transform(data), columns=data.columns)

# Normalisation Min-Max
def min_max_normalization(data):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Standardisation Z-score
def z_score_standardization(data):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Visualisation des histogrammes
def plot_histograms(data):
    st.write("Histograms of Numerical Columns in DataFrame:")
    fig, ax = plt.subplots(figsize=(10, 6))
    data.hist(bins=10, ax=ax)
    st.pyplot(fig)

# Visualisation des boîtes à moustaches
def plot_boxplots(data):
    st.write("Box Plots of Numerical Columns in DataFrame:")
    fig, ax = plt.subplots(figsize=(10, 6))
    data.boxplot(ax=ax)
    st.pyplot(fig)

# Méthode du coude pour K-Means
def elbow_method(data):
    st.write("Elbow Method For Optimal k:")
    k_values = range(1, 11)
    wcss = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, wcss, marker='o')
    ax.set_title('Elbow Method For Optimal k')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS')
    st.pyplot(fig)

# Clustering K-Means
def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data)
    return data

# Clustering DBSCAN
def dbscan_clustering(data, eps=0.3, min_samples=2):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    data['Cluster'] = clustering.labels_
    return data

# Visualisation des clusters en 2D
def plot_clusters_2d(data, params, cluster_column='Cluster'):
    st.write("2D Clustering Visualization:")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(data[params[0]], data[params[1]], c=data[cluster_column], cmap='viridis')
    ax.set_xlabel(params[0])
    ax.set_ylabel(params[1])
    fig.colorbar(scatter, ax=ax)
    st.pyplot(fig)

# Visualisation des clusters en 3D
def plot_clusters_3d(data, params, cluster_column='Cluster'):
    st.write("3D Clustering Visualization:")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[params[0]], data[params[1]], data[params[2]], c=data[cluster_column], cmap='viridis')
    ax.set_xlabel(params[0])
    ax.set_ylabel(params[1])
    ax.set_zlabel(params[2])
    fig.colorbar(scatter, ax=ax)
    st.pyplot(fig)


# Main 
def main():
    # Sidebar
    st.sidebar.title("DATA MINING PROJECT")
    st.sidebar.subheader("Members:")
    st.sidebar.write("Léa SELLAHANNADI")
    st.sidebar.write("Calvin NDOUMBE")
    st.sidebar.write("Faniry RAOBELINA")
    st.sidebar.subheader("BIA2")
    
    # Sidebar menu
    st.sidebar.subheader("SELECT PART")
    # Creation of a selectbox to be well-organized  
    menu = st.sidebar.selectbox("", ["Part I: Initial Data Exploration", "Part II: Data Pre-processing and Cleaning", "Part III: Visualization of the cleaned data", "Part IV: Clustering or Prediction", "Part V: Learning Evaluation"])

    if menu == "Part I: Initial Data Exploration":
        st.header("Part I: Initial Data Exploration")
        st.write("Data loading :")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        header = st.checkbox("Does the CSV file have a header ?", value=True)
        separator = st.text_input("Enter the separator:", value=',')
        
        if uploaded_file is not None:
            data = load_data(uploaded_file, header, separator)
            st.session_state['data'] = data  # Stocker les données dans la session
            st.write("Data description:")
            display_data(data)
            st.write("Statistical summary :")
            data_summary(data)

    elif menu == "Part II: Data Pre-processing and Cleaning":
        st.header("Part II: Data Pre-processing and Cleaning")
        
        if 'data' in st.session_state:
            data = st.session_state['data']
            data_modified = data.copy()  # Créer une copie des données initiales
            st.write("Initial data:")
            display_data(data)
            
            st.write("Options de prétraitement des données manquantes :")
            delete_columns = st.checkbox('Supprimer les colonnes avec des valeurs manquantes')
            delete_rows = st.checkbox('Supprimer les lignes avec des valeurs manquantes')
            fill_mean = st.checkbox('Remplacer les NaN par la moyenne')
            fill_median = st.checkbox('Remplacer les NaN par la médiane')
            fill_mode = st.checkbox('Remplacer les NaN par le mode')
            knn_impute = st.checkbox('Utiliser l\'imputation KNN')
            n_neighbors = st.slider('Nombre de voisins pour KNN', min_value=1, max_value=10, value=3) if knn_impute else 3

            if delete_columns:
                data_modified = delete_columns_with_nan(data_modified)
            if delete_rows:
                data_modified = delete_rows_with_nan(data_modified)
            if fill_mean:
                data_modified = nan_to_mean(data_modified)
            if fill_median:
                data_modified = nan_to_median(data_modified)
            if fill_mode:
                data_modified = nan_to_mode(data_modified)
            if knn_impute:
                data_modified = nan_to_knn_imputer(data_modified, n_neighbors=n_neighbors)

            st.write("Data after cleaning:")
            display_data(data_modified)
            st.write("Data summary after cleaning:")
            data_summary(data_modified)
        else:
            st.write("Veuillez d'abord charger les données dans la partie I.")


        
    elif menu == "Part III: Visualization of the cleaned data":
        st.header("Part III: Visualization of the cleaned data")

        #Pour garder le dataset de base et pouvoir faire les modifs
        if 'data' in st.session_state:
            data = st.session_state['data']
            data_modified = data.copy()  
            st.write("Initial data:")
            display_data(data)

        if menu == "Part III: Data Visualization":
            st.header("Part III: Data Visualization")
        
        if 'data' in st.session_state:
            data = st.session_state['data']
            st.write("Initial data:")
            display_data(data)

            st.write("Visualisation Options:")
            plot_hist = st.checkbox('Afficher les histogrammes')
            plot_box = st.checkbox('Afficher les boîtes à moustaches')

            if plot_hist:
                plot_histograms(data)
            if plot_box:
                plot_boxplots(data)
        else:
            st.write("Veuillez d'abord charger les données dans la partie I.")

    elif menu == "Part IV: Clustering or Prediction":
        st.header("Part IV: Clustering or Prediction")

        if 'data' in st.session_state:
            data = st.session_state['data']
            st.write("Initial data:")
            display_data(data)

            st.write("Clustering Options:")
            normalize = st.checkbox('Normaliser les données (Min-Max)')
            standardize = st.checkbox('Standardiser les données (Z-score)')

            if normalize:
                data = min_max_normalization(data)
            if standardize:
                data = z_score_standardization(data)

            st.write("Méthode de clustering:")
            clustering_method = st.selectbox('Choisissez une méthode de clustering', ['K-Means', 'DBSCAN'])

            if clustering_method == 'K-Means':
                st.write("K-Means Clustering:")
                elbow_method(data)
                optimal_k = st.slider('Nombre optimal de clusters (k)', min_value=1, max_value=10, value=3)
                data = kmeans_clustering(data, n_clusters=optimal_k)
                st.write("Clustered Data:")
                display_data(data)
                plot_2d = st.checkbox('Visualiser les clusters en 2D')
                plot_3d = st.checkbox('Visualiser les clusters en 3D')
                if plot_2d:
                    params_2d = st.multiselect('Choisissez deux paramètres pour la visualisation 2D', data.columns.tolist(), default=data.columns.tolist()[:2])
                    if len(params_2d) == 2:
                        plot_clusters_2d(data, params_2d)
                if plot_3d:
                    params_3d = st.multiselect('Choisissez trois paramètres pour la visualisation 3D', data.columns.tolist(), default=data.columns.tolist()[:3])
                    if len(params_3d) == 3:
                        plot_clusters_3d(data, params_3d)

            elif clustering_method == 'DBSCAN':
                st.write("DBSCAN Clustering:")
                eps = st.slider('Valeur de eps', min_value=0.1, max_value=5.0, value=0.5, step=0.1)
                min_samples = st.slider('Nombre minimum d\'échantillons', min_value=1, max_value=10, value=5)
                data = dbscan_clustering(data, eps=eps, min_samples=min_samples)
                st.write("Clustered Data:")
                display_data(data)
                plot_2d = st.checkbox('Visualiser les clusters en 2D')
                plot_3d = st.checkbox('Visualiser les clusters en 3D')
                if plot_2d:
                    params_2d = st.multiselect('Choisissez deux paramètres pour la visualisation 2D', data.columns.tolist(), default=data.columns.tolist()[:2])
                    if len(params_2d) == 2:
                        plot_clusters_2d(data, params_2d)
                if plot_3d:
                    params_3d = st.multiselect('Choisissez trois paramètres pour la visualisation 3D', data.columns.tolist(), default=data.columns.tolist()[:3])
                    if len(params_3d) == 3:
                        plot_clusters_3d(data, params_3d)

        else:
            st.write("Veuillez d'abord charger les données dans la partie I.")

    elif menu == "Part V: Learning Evaluation":
        st.header("Part V: Learning Evaluation")
        st.write("Cette partie pourrait inclure des méthodes d'évaluation des modèles d'apprentissage supervisé ou non supervisé.")
        # Placeholder for additional functionalities related to learning evaluation

    else:
        st.write("Veuillez d'abord charger les données dans la partie I.")

if __name__ == "__main__":
    main()