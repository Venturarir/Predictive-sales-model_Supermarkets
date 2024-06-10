import os
import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the models and data for the predictions
model_path = 'sales_volume_predictor.pkl'
preprocessor_path = 'preprocessor.pkl'
similarity_matrix_path = 'similarity_matrix.pkl'
product_data_path = 'product_data.pkl'

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
else:
    model = joblib.load(model_path)

    preprocessor = joblib.load(preprocessor_path)
    similarity_matrix = joblib.load(similarity_matrix_path)
    df = joblib.load(product_data_path)

    # Ensure that the similarity matrix has the correct dimensions
    preprocessed_data = preprocessor.transform(df.drop(columns=['Product ID']))

    # Function to find the three most similar items for a given product
    def get_most_similar_items(input_data, n=3):
    
        input_data_encoded = preprocessor.transform(input_data)

        input_similarity = cosine_similarity(input_data_encoded, preprocessed_data)
        # Sort the scores in descending order and get the indices of the most similar products
        similar_indices = input_similarity.argsort()[0][::-1][:n]
        # Get the details of the most similar products
        similar_products = df.iloc[similar_indices]
        return similar_products

    # Streamlit app
    st.set_page_config(page_title="Sales Volume Predictor", page_icon="📈", layout="wide")

    # Custom to style the app
    st.markdown("""
        <style>
            .main {
                background-color: #f5f5f5;
            }
            .stButton>button {
                color: white;
                background-color: #4CAF50;
                border: none;
                padding: 10px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                transition-duration: 0.4s;
                cursor: pointer;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            .title {
                font-size: 2em;
                font-weight: bold;
                color: #4CAF50;
            }
            .subtitle {
                font-size: 1.5em;
                font-weight: bold;
                color: #333;
            }
            .header {
                font-size: 1.2em;
                font-weight: bold;
                color: #666;
            }
        </style>
    """, unsafe_allow_html=True)

    # Header and Logo
    image_path = '/Users/ventura/Desktop/ESADE BBA/MSC Analytics/TERM 3/Prototyping with AI/Assigments/Supermarket_image.jpg'
    if os.path.exists(image_path):
        st.image(image_path, width=150)
    else:
        st.warning(f"Image file not found: {image_path}")

    st.markdown('<div class="title">Sales Volume Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict the sales volume based on product details</div>', unsafe_allow_html=True)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Enter Product Details", "Batch Prediction"])

    if page == "Enter Product Details":
        # Collect user input for all features
        st.markdown('<div class="header">Enter Product Details</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            product_position = st.selectbox('Product Position', ['Aisle', 'End-cap', 'Front of Store'])
            price = st.slider('Price', min_value=0.0, max_value=100.0, step=0.01, value=20.0)
            competitors_price = st.slider("Competitor's Price", min_value=0.0, max_value=100.0, step=0.01, value=20.0)
            promotion = st.selectbox('Promotion', ['Yes', 'No'])
        with col2:
            foot_traffic = st.selectbox('Foot Traffic', ['Low', 'Medium', 'High'])
            consumer_demographics = st.selectbox('Consumer Demographics', ['Families', 'Seniors', 'Young adults', 'College students'])
            product_category = st.selectbox('Product Category', ['Clothing', 'Electronics', 'Food'])
            seasonal = st.selectbox('Seasonal', ['Yes', 'No'])

        description = st.text_area('Product Description', 'Enter a brief description of the product')

        # Create a dictionary from the input
        input_data = {
            'Description': description,
            'Product Position': product_position,
            'Price': price,
            "Competitor's Price": competitors_price,
            'Promotion': promotion,
            'Foot Traffic': foot_traffic,
            'Consumer Demographics': consumer_demographics,
            'Product Category': product_category,
            'Seasonal': seasonal
        }

        # Convert the dictionary to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Make prediction
        if st.button('Predict Sales Volume'):
            prediction = model.predict(input_df.drop(columns=['Description']))
            st.success(f'Predicted Sales Volume: {prediction[0]:.2f}')

            # Add the predicted sales volume to the input data
            input_df['Sales Volume'] = prediction[0]

            # Find similar products using the input data including the predicted sales volume
            similar_products = get_most_similar_items(input_df)

            # Display similar products with details and emojis
            st.markdown("### Similar Products")
            for index, product in similar_products.iterrows():
                st.markdown(f"""
                **Product ID**: {product['Product ID']}  
                **Product Position**: {product['Product Position']} 🏬  
                **Price**: ${product['Price']:.2f} 💵  
                **Competitor's Price**: ${product["Competitor's Price"]:.2f} 🛒  
                **Promotion**: {product['Promotion']} 🎉  
                **Foot Traffic**: {product['Foot Traffic']} 🚶‍♂️  
                **Consumer Demographics**: {product['Consumer Demographics']} 👥  
                **Product Category**: {product['Product Category']} 🏷️  
                **Seasonal**: {product['Seasonal']} 🍂  
                **Sales Volume**: {product['Sales Volume']} 📊  
                """)

    elif page == "Batch Prediction":
        # Batch Prediction using File Upload
        st.markdown('<div class="header">Batch Prediction</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=["csv"])

        if uploaded_file is not None:
            batch_data = pd.read_csv(uploaded_file)

            # Ensure the file contains the required columns
            required_columns = ['Product Position', 'Price', "Competitor's Price", 'Promotion', 'Foot Traffic', 'Consumer Demographics', 'Product Category', 'Seasonal']
            if all(column in batch_data.columns for column in required_columns):
                predictions = model.predict(batch_data[required_columns])
                batch_data['Predicted Sales Volume'] = predictions
                st.write(batch_data)
            else:
                st.error(f"Uploaded file must contain the following columns: {required_columns}")
