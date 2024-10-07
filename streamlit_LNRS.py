import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Your Hugging Face token
token = "your_huggingface_token"  # Replace with your actual Hugging Face token

# Load three different sentiment analysis models
sentiment_model_1 = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest", 
    use_auth_token=token
)
sentiment_tokenizer_1 = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest", 
    use_auth_token=token
)
sentiment_pipeline_1 = pipeline("sentiment-analysis", model=sentiment_model_1, tokenizer=sentiment_tokenizer_1)

sentiment_model_2 = AutoModelForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment", 
    use_auth_token=token
)
sentiment_tokenizer_2 = AutoTokenizer.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment", 
    use_auth_token=token
)
sentiment_pipeline_2 = pipeline("sentiment-analysis", model=sentiment_model_2, tokenizer=sentiment_tokenizer_2)

sentiment_model_3 = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english", 
    use_auth_token=token
)
sentiment_tokenizer_3 = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english", 
    use_auth_token=token
)
sentiment_pipeline_3 = pipeline("sentiment-analysis", model=sentiment_model_3, tokenizer=sentiment_tokenizer_3)

# Load the sarcasm detection model with the token
sarcasm_model = AutoModelForSequenceClassification.from_pretrained(
    "jkhan447/sarcasm-detection-Bert-base-uncased", 
    use_auth_token=token
)
sarcasm_tokenizer = AutoTokenizer.from_pretrained(
    "jkhan447/sarcasm-detection-Bert-base-uncased", 
    use_auth_token=token
)
sarcasm_pipeline = pipeline("text-classification", model=sarcasm_model, tokenizer=sarcasm_tokenizer)

# Map the sarcasm detection labels to descriptive terms
def map_sarcasm_label(label):
    if label == "LABEL_0":
        return "Not Sarcastic"
    elif label == "LABEL_1":
        return "Sarcastic"
    else:
        return "Unknown"

# Majority voting for sentiment analysis
def majority_vote_sentiment(sentiment_results):
    sentiment_votes = {"positive": 0, "negative": 0, "neutral": 0}

    for result in sentiment_results:
        sentiment_label = result[0]["label"].lower()
        if sentiment_label in sentiment_votes:
            sentiment_votes[sentiment_label] += 1
    
    # Get the sentiment with the highest votes
    majority_sentiment = max(sentiment_votes, key=sentiment_votes.get)
    return majority_sentiment

# Streamlit App Interface
st.title("Sentiment and Sarcasm Detection App")

# Text input from the user
input_text = st.text_input("Enter a sentence for sentiment and sarcasm analysis:")

if input_text:
    # Get sentiment results from the three models
    sentiment_result_1 = sentiment_pipeline_1(input_text)
    sentiment_result_2 = sentiment_pipeline_2(input_text)
    sentiment_result_3 = sentiment_pipeline_3(input_text)

    # Aggregate the results and apply majority voting
    final_sentiment = majority_vote_sentiment([sentiment_result_1, sentiment_result_2, sentiment_result_3])

    # Get sarcasm detection result and map the label
    sarcasm_result = sarcasm_pipeline(input_text)
    mapped_sarcasm_result = {
        'label': map_sarcasm_label(sarcasm_result[0]['label']),
        'score': sarcasm_result[0]['score']
    }

    # Override sentiment to negative if sarcasm is detected
    if mapped_sarcasm_result['label'] == "Sarcastic":
        final_sentiment = "negative"

    # Display the results in Streamlit
    st.write(f"Final Sentiment: **{final_sentiment.capitalize()}**")
    st.write(f"Sarcasm Detection: **{mapped_sarcasm_result['label']}**")

# Button to clear the input and reset
if st.button("Clear"):
    st.experimental_rerun()
