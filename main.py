import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

def load_df(file, header, separator):
    if header:
        df = pd.read_csv(file, sep=separator)
    else:
        df = pd.read_csv(file, sep=separator, header=None)
    return df

def nan_to_remove_columns(df):
    return df.dropna(axis=1)

def remove_nan(df):
    return df.dropna()

def nan_to_mean(df, numerical_columns):
    df[numerical_columns] = df[numerical_columns].apply(lambda x: x.fillna(x.mean()), axis=0)
    return df

def nan_to_mode(df, numerical_columns):
    df[numerical_columns] = df[numerical_columns].apply(lambda x: x.fillna(x.mode()[0]), axis=0)
    return df

def nan_to_med(df, numerical_columns):
    df[numerical_columns] = df[numerical_columns].apply(lambda x: x.fillna(x.median()), axis=0)
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

def decimal_place_normalization(df):
    return df.round(decimals=2)

def standard_deviation_normalization(df, numerical_columns):
    scaler = StandardScaler(with_mean=False)
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df

def plot_knn_distances(df, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(df)
    distances, indices = nbrs.kneighbors(df)
    distances = np.sort(distances[:, k-1], axis=0)
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('k-NN Distance Graph')
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel(f'{k}-NN Distance')
    st.pyplot(plt)

def plot_linear_regression(df, x_column, y_column):
    X = df[[x_column]].values.reshape(-1, 1)
    y = df[y_column].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual data')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Regression line')
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'Linear Regression: {x_column} vs {y_column}')
    plt.legend()
    st.pyplot(plt)

def plot_roc_curve(y_test, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)

def main():
    st.title("INTERACTIVE WEB APPLICATION🖥️")
    st.write("Analyze, clean, and visualize your data effectively with our advanced techniques!")

    st.sidebar.title("DATA MINING PROJECT BIA2")
    st.sidebar.markdown("### Calvin NDOUMBE - Faniry RAOBELINA - Léa SELLAHANNADI")

    st.sidebar.header("Initial Data Exploration 🔍")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file: ", type=["csv","data"])

    if uploaded_file:
        header = st.sidebar.checkbox("Does the file have a header?", value=True)
        separator = st.sidebar.text_input("Type of separation: ", value=",")
        df = load_df(uploaded_file, header, separator)

        st.title("Data Description 📋")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Preview the first rows:")
            st.write(df.head())
        with col2:
            st.write("Preview the last rows:")
            st.write(df.tail())

        st.title("Statistical Summary 🔢")

        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")

        col_types = pd.DataFrame(df.dtypes, columns=['Data Type']).reset_index()
        col_types.columns = ['Column Name', 'Data Type']

        missing_values = df.isnull().sum().reset_index()
        missing_values.columns = ['Column Name', 'Missing Values']
        summary_df = pd.merge(col_types, missing_values, on='Column Name')

        st.write("Column names, data types, and missing values per column: ")
        st.write(summary_df)

        st.write("Descriptive statistics: ")
        st.write(df.describe())

        numerical_columns = [i for i in df.columns if df[i].dtype in ['int64', 'float64']]
        non_numerical_columns = [i for i in df.columns if df[i].dtype not in ['int64', 'float64']]

        st.sidebar.subheader("Data Pre-processing and Cleaning 🧼")
        missing_values_option = st.sidebar.selectbox("Choose a method to handle missing values:",
                                                     ["Remove Columns","Remove Rows", "Fill with Mean", "Fill with Mode", "Fill with Median", "KNN Imputer"])
        
        if missing_values_option == "Remove Columns":
            df = nan_to_remove_columns(df)
        elif missing_values_option == "Remove Rows":
            df = remove_nan(df)
        elif missing_values_option == "Fill with Mean":
            df = nan_to_mean(df, numerical_columns)
        elif missing_values_option == "Fill with Mode":
            df = nan_to_mode(df, numerical_columns)
        elif missing_values_option == "Fill with Median":
            df = nan_to_med(df, numerical_columns)
        elif missing_values_option == "KNN Imputer":
            df = nan_to_knn_imputer(df, numerical_columns)
      
        st.title("Managing Missing Values 🧹")
        st.write("Dataframe after handling the missing values: ")
        st.write(df.head())

        st.sidebar.subheader("Normalization ⚖️")
        normalization_option = st.sidebar.selectbox("Choose a method: ",
                                                    ["None", "Min-Max Normalization", "Z-score Standardization", "Decimal Place Normalization", "Standard Deviation Normalization"])
        if normalization_option == "Min-Max Normalization":
            df = min_max_normalization(df, numerical_columns)
        elif normalization_option == "Z-score Standardization":
            df = zscore_standardization(df, numerical_columns)
        elif normalization_option == "Decimal Place Normalization":
            df = decimal_place_normalization(df)
        elif normalization_option == "Standard Deviation Normalization":
            df = standard_deviation_normalization(df, numerical_columns)

        st.title("Normalizing the Data 📏")
        st.write("Dataframe after data normalization: ")
        st.write(df.head())

        st.sidebar.subheader("Visualization 📊")
        selected_columns_hist = st.sidebar.multiselect("Select columns for histograms:", numerical_columns)
        selected_columns_box = st.sidebar.multiselect("Select columns for box plots:", numerical_columns)
        selected_columns_bar = st.sidebar.multiselect("Select columns for bar plots:", non_numerical_columns)

        st.title("Visualization of the Cleaned Data 📊")

        col1, col2 = st.columns(2)
        
        with col1:
            if selected_columns_hist:
                st.write("Histograms:")
                fig, axes = plt.subplots(len(selected_columns_hist), 1, figsize=(8, 6 * len(selected_columns_hist)))
                if len(selected_columns_hist) == 1:
                    axes = [axes]
                for ax, column in zip(axes, selected_columns_hist):
                    df[column].plot(kind='hist', ax=ax, title=column, color='skyblue', edgecolor='black', bins=20)
                    ax.set_xlabel(column)
                    ax.set_ylabel('Frequency')
                    ax.grid(True)
                    ax.set_title(f'Histogram of {column}', fontsize=10)
                st.pyplot(fig)
            else:
                st.write("Please select columns for histograms.")
        
        with col2:
            if selected_columns_box:
                st.write("Box Plots:")
                fig, ax = plt.subplots()
                df[selected_columns_box].boxplot(ax=ax, vert=False, patch_artist=True)
                plt.title("Box plots")
                ax.grid(True)
                ax.set_title("Box plots", fontsize=10)
                st.pyplot(fig)
            else:
                st.write("Please select columns for box plots.")

            if selected_columns_bar:
                st.write("Bar Plots:")
                fig, axes = plt.subplots(len(selected_columns_bar), 1, figsize=(8, 6 * len(selected_columns_bar)))
                if len(selected_columns_bar) == 1:
                    axes = [axes]
                for ax, column in zip(axes, selected_columns_bar):
                    df[column].value_counts().plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
                    ax.set_title(f'Bar Plot of {column}', fontsize=10)
                    ax.set_xlabel(column)
                    ax.set_ylabel('Count')
                    ax.grid(True)
                st.pyplot(fig)
            else:
                st.write("Please select columns for bar plots.")

        st.sidebar.subheader("Clustering or Prediction")
        task_option = st.sidebar.selectbox("Choose a task:", ["Clustering", "Prediction"])

        if task_option == "Clustering":
            st.sidebar.subheader("Clustering 🌀")
            clustering_option = st.sidebar.selectbox("Choose a clustering method:", ["KMeans", "DBSCAN"])

            st.title("Modeling the Data 🔧")
            if clustering_option == "KMeans":
                k = st.sidebar.slider("Select number of clusters for KMeans:", 1, 10, 3)
                kmeans = KMeans(n_clusters=k)
                df['Cluster'] = kmeans.fit_predict(df[numerical_columns])
                
                st.title("Learning Evaluation 📚")
                k_values = range(1, 11)
                wcss = []

                for k in k_values:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(df[numerical_columns])
                    wcss.append(kmeans.inertia_)

                fig, ax = plt.subplots()
                ax.plot(k_values, wcss, marker='o')
                plt.title('Elbow Method for optimal k')
                plt.xlabel('Number of clusters')
                plt.ylabel('WCSS')
                plt.xticks(k_values)
                st.pyplot(fig)

            elif clustering_option == "DBSCAN":
                st.sidebar.write("Choose `eps` value by examining the k-NN distance graph.")
                k = st.sidebar.slider("Select number of nearest neighbors (k)", 1, 10, 4)
                plot_knn_distances(df[numerical_columns], k=k)
                eps = st.sidebar.slider("Select epsilon for DBSCAN", 0.1, 1.0, 0.3)
                min_samples = st.sidebar.slider("Select min_samples for DBSCAN", 1, 10, 2)
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                df['Cluster'] = dbscan.fit_predict(df[numerical_columns])

            st.title("Visualization of Clusters 📊")
            selected_columns_2d = st.sidebar.multiselect("Select two columns for 2D visualization:", numerical_columns, default=numerical_columns[:2])
            selected_columns_3d = st.sidebar.multiselect("Select three columns for 3D visualization:", numerical_columns, default=numerical_columns[:3])
            
            if len(selected_columns_2d) == 2:
                fig, ax = plt.subplots()
                scatter = ax.scatter(df[selected_columns_2d[0]], df[selected_columns_2d[1]], c=df['Cluster'], cmap='viridis')
                legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                plt.title('2D Clustering Visualization')
                plt.xlabel(selected_columns_2d[0])
                plt.ylabel(selected_columns_2d[1])
                st.pyplot(fig)
            else:
                st.write("Please select exactly two columns for 2D visualization: ")

            if len(selected_columns_3d) == 3:
                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(df[selected_columns_3d[0]], df[selected_columns_3d[1]], df[selected_columns_3d[2]], 
                                    c=df['Cluster'], cmap='viridis', marker='o')
                ax.set_title('3D Clustering Visualization')
                ax.set_xlabel(selected_columns_3d[0])
                ax.set_ylabel(selected_columns_3d[1])
                ax.set_zlabel(selected_columns_3d[2])
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Cluster')
                st.pyplot(fig)
            else:
                st.write("Please select exactly three columns for 3D visualization.")

            # Cluster statistics
            st.title("Cluster Statistics 📊")
            cluster_stats = df['Cluster'].value_counts().sort_index()
            st.write(cluster_stats)
            if clustering_option == "KMeans":
                cluster_centers = kmeans.cluster_centers_
            elif clustering_option == "DBSCAN":
                unique_labels = set(df['Cluster'])
                for label in unique_labels:
                    if label != -1:
                        cluster_density = len(df[df['Cluster'] == label]) / np.sum(df['Cluster'] == label)
                        st.write(f"Density of cluster {label}: {cluster_density}")

        elif task_option == "Prediction":
            st.sidebar.subheader("Prediction 📈")
            problem_type = st.sidebar.selectbox("Select the problem type:", ["Classification", "Regression"])
            target_column = st.sidebar.selectbox("Select the target column:", numerical_columns + ['whether he/she donated blood in March 2007'])
            
            st.title("Modeling the Data 🔧")
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Ensure that only numeric data is used for prediction
            X = X.select_dtypes(include=[np.number])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if problem_type == "Classification":
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_prob = model.predict_proba(X_test)[:, 1]
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy}")

                st.write("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
                plt.title('Confusion Matrix Heatmap')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(plt)

                st.write("Classification Report")
                st.write(classification_report(y_test, y_pred))

                st.title("ROC Curve 📈")
                plot_roc_curve(y_test, y_pred_prob)

                st.title("Prediction Scatter Plot")
                x_column = st.sidebar.selectbox("Select the x-axis column for scatter plot:", X.columns)
                y_column = st.sidebar.selectbox("Select the y-axis column for scatter plot:", [target_column])

                if x_column and y_column:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(X_test[x_column], y_test, color='blue', label='Actual data')
                    plt.scatter(X_test[x_column], y_pred, color='red', marker='x', label='Predicted data')
                    plt.xlabel(x_column)
                    plt.ylabel(y_column)
                    plt.title(f'Actual vs Predicted: {x_column} vs {y_column}')
                    plt.legend()
                    st.pyplot(plt)

            elif problem_type == "Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"Mean Squared Error: {mse}")

                st.write("Actual vs Predicted values")
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                plt.title('Actual vs Predicted values')
                st.pyplot(fig)
                
                st.sidebar.subheader("Linear Regression Plot 📈")
                x_column = st.sidebar.selectbox("Select the x-axis column for Linear Regression plot:", X.columns)
                y_column = st.sidebar.selectbox("Select the y-axis column for Linear Regression plot:", [target_column])
                if x_column and y_column:
                    plot_linear_regression(df, x_column, y_column)

if __name__ == "__main__":
    main()
