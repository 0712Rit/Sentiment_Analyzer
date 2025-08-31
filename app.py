import streamlit as st
import pickle
import re
import time
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# --- LOAD MODELS AND VECTORIZER ---
# Use st.cache_resource to load the model and vectorizer only once.
@st.cache_resource
def load_assets():
    """Loads the pre-trained model and vectorizer from disk."""
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model or vectorizer file not found. Please make sure 'model.pkl' and 'vectorizer.pkl' are in the same directory.")
        return None, None

model, vectorizer = load_assets()
port_stem = PorterStemmer()

# --- TEXT PREPROCESSING ---
def stemming(content):
    """
    Performs stemming and stopword removal on the input text.
    This function should be identical to the one used for training.
    """
    # Remove non-alphabetic characters
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    # Convert to lowercase and split into words
    stemmed_content = stemmed_content.lower().split()
    # Stem words and remove English stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    # Join words back into a single string
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# --- PREDICTION FUNCTION ---
def predict_sentiment(text):
    """
    Predicts the sentiment and confidence score for the input text.
    """
    if model is None or vectorizer is None:
        return None, None
    stemmed_text = stemming(text)
    text_vector = vectorizer.transform([stemmed_text])
    prediction = model.predict(text_vector)
    prediction_prob = model.predict_proba(text_vector)
    return prediction[0], prediction_prob

# --- MAIN APP ---
def main():
    """Main function to run the Streamlit app."""
    # --- PAGE CONFIGURATION ---
    st.set_page_config(
        page_title="Sentiment Analyzer",
        page_icon="🐦",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # HEADER
    st.title("🐦Twitter Sentiment Analyzer")
    st.markdown(
        "Analyze the sentiment of any text with my AI-powered tool. "
        "Just type or paste your text below and see the magic happen!"
    )
    st.markdown("---")

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("📝 About")
        st.info(
            "This app uses a Logistic Regression model to classify text sentiment as "
            "**Positive** or **Negative**. The model was trained on a Twitter dataset."
        )
        st.header("💡 Example Tweets")
        example_tweets = [
            "I love this new phone, the camera is amazing!",
            "The weather is so gloomy today, I feel sad.",
            "Just had the best meal ever at that new restaurant.",
            "My flight was delayed for hours, what a terrible experience."
        ]
        # Use a selectbox for a dropdown menu of examples
        selected_example = st.selectbox(
            "Choose an example to analyze:",
            options=[""] + example_tweets,
            index=0,
            format_func=lambda x: "Select an example..." if x == "" else x
        )


    # --- USER INPUT ---
    # The text area's value is updated if an example is selected
    user_input = st.text_area(
        "Your text for analysis:",
        value=selected_example,
        placeholder="Enter text here...",
        height=150,
        key="text_area"
    )

    # --- ANALYSIS BUTTON ---
    if st.button("Analyze Sentiment ✨"):
        if user_input:
            with st.spinner('Analyzing your text...'):
                time.sleep(1) # Simulate processing time for a better UX
                prediction, prediction_prob = predict_sentiment(user_input)

            if prediction is not None:
                st.markdown("---")
                st.subheader("📊 Analysis Result")

                # Display result in columns for a cleaner layout
                col1, col2 = st.columns(2)

                with col1:
                    if prediction == 1:
                        st.markdown(
                            """
                            <div style="padding: 20px; border-radius: 10px; background-color: #28a745; color: white; text-align: center;">
                                <h3 style="color: white;">Positive 😊</h3>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            """
                            <div style="padding: 20px; border-radius: 10px; background-color: #dc3545; color: white; text-align: center;">
                                <h3 style="color: white;">Negative 😞</h3>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                with col2:
                    confidence = prediction_prob.max() * 100
                    st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                    st.progress(int(confidence))

        else:
            st.warning("🤔 Please enter some text to analyze.")

    # --- FOOTER ---
    st.markdown("---")
    st.markdown("Made with ❤️ by RITAM using Streamlit")


if __name__ == '__main__':
    # Ensure NLTK stopwords are downloaded
    import nltk
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    main()
