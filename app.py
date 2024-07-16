# Importation of the useful libraries
import streamlit as st
import pandas as pd

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
            
            delete_columns = st.checkbox('Supprimer les colonnes avec des valeurs manquantes')
            delete_rows = st.checkbox('Supprimer les lignes avec des valeurs manquantes')

            if delete_columns:
                data_modified = delete_columns_with_nan(data_modified)

            if delete_rows:
                data_modified = delete_rows_with_nan(data_modified)

            # Afficher le tableau modifié une seule fois
            if delete_columns or delete_rows:
                st.write("Tableau après suppression des valeurs manquantes :")
                display_data(data_modified)
                st.write("Data summary after cleaning:")
                data_summary(data_modified)
            else:
                st.write("Aucune modification n'a été appliquée.")

        else:
            st.write("Veuillez d'abord charger les données dans la partie I.")
        
    elif menu == "Part III: Data Pre-processing and Cleaning":
        st.header("Part III: Data Pre-processing and Cleaning")

        #Pour garder le dataset de base et pouvoir faire les modifs
        if 'data' in st.session_state:
            data = st.session_state['data']
            data_modified = data.copy()  
            st.write("Initial data:")
            display_data(data)

        
        else:
            st.write("Veuillez d'abord charger les données dans la partie I.")

if __name__ == "__main__":
    main()
