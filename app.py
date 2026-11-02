
import streamlit as st
import joblib

# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

# --------------------------------
# Load Model and TF-IDF Vectorizer
# --------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("log_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return model, vectorizer


log_model, tfidf_vectorizer = load_model()

# --------------------------------
# Application UI
# --------------------------------

st.title("💬 Sentiment Analyzer")

st.write(
    "Analyze the sentiment of your text "
    "using Machine Learning and NLP."
)

st.markdown("---")

user_input = st.text_area(
    "Enter your text below:",
    placeholder="Type something here...",
    height=150
)

# --------------------------------
# Sentiment Prediction
# --------------------------------

if st.button("Analyze Sentiment", use_container_width=True):

    if not user_input.strip():

        st.warning("Please enter some text!")

    else:

        # Convert input text into TF-IDF features
        text_vector = tfidf_vectorizer.transform([user_input])

        # Predict sentiment
        prediction = log_model.predict(text_vector)[0]

        # Get prediction probabilities
        probabilities = log_model.predict_proba(text_vector)[0]

        confidence = max(probabilities) * 100

        # Display prediction
        st.markdown("---")
        st.subheader("Prediction Result")

        st.success(f"Predicted Sentiment: {prediction}")

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        # Display probabilities of all classes
        st.subheader("Sentiment Probabilities")

        for sentiment, probability in zip(
            log_model.classes_,
            probabilities
        ):

            st.write(
                f"**{sentiment}:** {probability * 100:.2f}%"
            )

            st.progress(float(probability))

# --------------------------------
# Footer
# --------------------------------

st.markdown("---")

st.caption(
    "Built using Python, Streamlit, TF-IDF "
    "and Logistic Regression."
)
