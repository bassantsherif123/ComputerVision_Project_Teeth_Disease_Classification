import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Load Model
model =  tf.keras.models.load_model('Model/Modified_InceptionResNetV2_Model_Teeth_disease_classification.keras')

diseases = [
    'Canker Sores (CaS)',
    'Cold Sores (CoS)',
    'Gum Disease (Gum)',
    'Mouth Cancer (MC)',
    'Oral Cancer (OC)', 
    'Oral Lichen Planus (OLP)',
    'Oral Thrush (OT)'
    ]

st.set_page_config(page_title='Teeth Disease Classifier')

st.title('🩺 Teeth Disease Classification')
uploaded_file = st.file_uploader('Please Upload an Image 📸', type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)
    img_array = img_array / 255.0

    st.markdown(
        f'<h4 style=\'text-align: center; font-weight: bold;\'>🚀 This Classifer is using a modified InceptionResNetV2 Model</h4>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<p style=\'text-align: center; font-style: italic;\'>⌚This may take a few seconds, please wait⌚</p>',
        unsafe_allow_html=True
    )

    # Make Predictions
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = diseases[predicted_index]
    confidence = np.max(predictions) * 100

    st.markdown(
        f'<h2 style=\'text-align: center; font-weight: bold;\'>Prediction: {predicted_class} </br> Confidence: {confidence:.2f} %</h2>',
        unsafe_allow_html=True
    )

    st.markdown(       
        f'</br><p style=\'text-align: center; font-style: italic;\'>✨ For more details about class distributions, please check the sidebar ✨</p>',
        unsafe_allow_html=True
    )

    # SideBar
    # 1. Pie Chart
    st.sidebar.subheader('📊 Class Probability Pie Chart')
 
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts = ax.pie(predictions[0], startangle=90)
    ax.axis('equal')

    ax.legend(
        wedges, diseases,
        title='Teeth Diseases',
        loc='lower left',
        bbox_to_anchor=(-0.15, -0.15),
        fontsize=9
    )

    ax.text(1.2, 0, f'Prediction:\n{predicted_class}\nConfidence: {confidence:.2f}%',
            fontsize=12, fontweight='bold', ha='left', va='center')

    st.sidebar.pyplot(fig)

    # 2. Confidence Scores Dataframe
    confidence_dict = {diseases[i]: float(predictions[0][i] * 100) for i in range(len(diseases))}

    st.sidebar.subheader('🔍 Class Confidence Scores')
    df = pd.DataFrame(list(confidence_dict.items()), columns=['Class', 'Percentage'])
    df.set_index('Class', inplace=True)
    st.sidebar.dataframe(df, use_container_width=True)