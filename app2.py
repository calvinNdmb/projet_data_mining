import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error

# Part I: Initial Data Exploration

st.title("DATA MINING PROJECT BIA2")
st.header("Léa Sellahannadi - Calvin Ndoumbe - Faniry Raobelina")

# Emoji Loop in Header
st.header("Data Loading 🚀")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    delimiter = st.text_input("Delimiter (e.g., ',', ';', '|')", value=',', key="delimiter")
    header = st.number_input("Header row (0-indexed)", min_value=0, value=0, key="header")
    
    try:
        data = pd.read_csv(uploaded_file, delimiter=delimiter, header=header)
        st.success("File loaded successfully!")
        
        # Data Description
        st.header("Data Description 📊")
        st.subheader("Preview of Data")
        st.write("First 5 rows of the data:")
        st.write(data.head())
        st.write("Last 5 rows of the data:")
        st.write(data.tail())
        
        # Column Data Types
        st.subheader("Column Data Types")
        col_types = pd.DataFrame(data.dtypes, columns=['Data Type']).reset_index()
        col_types.columns = ['Column Name', 'Data Type']
        st.write(col_types)
        
        # Statistical Summary
        st.header("Statistical Summary 📈")
        st.write(f"Number of rows: {data.shape[0]}")
        st.write(f"Number of columns: {data.shape[1]}")
        st.write("Column Names:")
        st.write(list(data.columns))
        
        missing_values = data.isnull().sum()
        st.write("Number of Missing Values per Column:")
        st.write(missing_values)
        
        st.write("Descriptive Statistics of the Data:")
        st.write(data.describe())
        
        # Part II: Data Pre-processing and Cleaning
        
        st.header("Data Pre-processing and Cleaning 🧹")
        
        # Radio options for handling missing values
        st.subheader("Handle Missing Values ❓")
        missing_value_method = st.radio(
            "Select method to handle missing values",
            [
                "None",
                "Remove columns with missing values 🗑️", 
                "Fill with mean 📊", 
                "Fill with median 🧮", 
                "Fill with mode 🛠️", 
                "Remove rows with missing values 🧹", 
                "Fill with KNN imputer 🤖"
            ],
            key="missing_values"
        )

        # Apply selected method to handle missing values
        if missing_value_method == "Remove columns with missing values 🗑️":
            data = data.dropna(axis=1)
        elif missing_value_method == "Fill with mean 📊":
            data = data.fillna(data.mean())
        elif missing_value_method == "Fill with median 🧮":
            data = data.fillna(data.median())
        elif missing_value_method == "Fill with mode 🛠️":
            for column in data.select_dtypes(include=['object']).columns:
                data[column].fillna(data[column].mode()[0], inplace=True)
        elif missing_value_method == "Remove rows with missing values 🧹":
            data = data.dropna()
        elif missing_value_method == "Fill with KNN imputer 🤖":
            imputer = KNNImputer(n_neighbors=3)
            data[data.columns] = imputer.fit_transform(data)
        
        st.write("Data after handling missing values:")
        st.write(data)
        
        # Radio options for normalization
        st.subheader("Normalize Data 📏")
        normalization_method = st.radio(
            "Select method to normalize data",
            [
                "None",
                "Min-Max normalization 🔄", 
                "Z-score standardization 📉"
            ],
            key="normalization"
        )
        
        # Apply selected method for normalization
        if normalization_method == "Min-Max normalization 🔄":
            scaler = MinMaxScaler()
            data[data.columns] = scaler.fit_transform(data)
        elif normalization_method == "Z-score standardization 📉":
            scaler = StandardScaler()
            data[data.columns] = scaler.fit_transform(data)
        
        st.write("Data after normalization:")
        st.write(data)
        
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.info("Please upload a CSV file to get started.")

# Part III: Visualization of the cleaned data

st.header("Visualization of the Cleaned Data 📊")

if 'data' in locals() or 'data' in globals():
    st.subheader("Select a feature to visualize")
    
    columns = data.columns.tolist()
    selected_column = st.selectbox("Choose a column", columns)
    
    if selected_column:
        st.subheader(f"Visualizations for {selected_column}")

        # Histogram
        st.write("Histogram 📈")
        fig, ax = plt.subplots()
        sns.histplot(data[selected_column], kde=True, ax=ax)
        ax.set_title(f"Histogram of {selected_column}")
        st.pyplot(fig)

        # Box Plot
        st.write("Box Plot 📦")
        fig, ax = plt.subplots()
        sns.boxplot(x=data[selected_column], ax=ax)
        ax.set_title(f"Box Plot of {selected_column}")
        st.pyplot(fig)
else:
    st.info("Please upload and preprocess the data to visualize.")

# Part IV: Clustering or Prediction

st.header("Clustering or Prediction 🔍")

if 'data' in locals() or 'data' in globals():
    task = st.radio(
        "Select a task:",
        ("Clustering", "Prediction"),
        key="task_selection"
    )

    if task == "Clustering":
        st.subheader("Clustering Algorithms")
        clustering_algo = st.selectbox("Select Clustering Algorithm", ["K-Means", "DBSCAN"])

        if clustering_algo == "K-Means":
            st.write("K-Means Clustering")
            num_clusters = st.number_input("Number of clusters", min_value=2, value=3)
            kmeans = KMeans(n_clusters=num_clusters)
            data['Cluster'] = kmeans.fit_predict(data.select_dtypes(include=[float, int]))
            st.write(data)
            fig, ax = plt.subplots()
            ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data['Cluster'], cmap='viridis')
            ax.set_title("K-Means Clustering")
            st.pyplot(fig)

        elif clustering_algo == "DBSCAN":
            st.write("DBSCAN Clustering")
            eps = st.number_input("Epsilon", min_value=0.1, value=0.5)
            min_samples = st.number_input("Minimum samples", min_value=1, value=5)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            data['Cluster'] = dbscan.fit_predict(data.select_dtypes(include=[float, int]))
            st.write(data)
            fig, ax = plt.subplots()
            ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data['Cluster'], cmap='viridis')
            ax.set_title("DBSCAN Clustering")
            st.pyplot(fig)

    elif task == "Prediction":
        st.subheader("Prediction Algorithms")
        prediction_algo = st.selectbox("Select Prediction Algorithm", ["Linear Regression", "Logistic Regression"])

        target_column = st.selectbox("Select target column", data.columns)

        if prediction_algo == "Linear Regression":
            st.write("Linear Regression")
            X = data.drop(columns=[target_column])
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            st.write(f"Mean Squared Error: {mse}")
            st.write("Predictions vs Actual:")
            results = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
            st.write(results)

        elif prediction_algo == "Logistic Regression":
            st.write("Logistic Regression")
            X = data.drop(columns=[target_column])
            y = data[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            report = classification_report(y_test, predictions, output_dict=True)
            st.write("Classification Report:")
            st.write(pd.DataFrame(report).transpose())

else:
    st.info("Please upload and preprocess the data to perform clustering or prediction.")


# Part V: Learning Evaluation

