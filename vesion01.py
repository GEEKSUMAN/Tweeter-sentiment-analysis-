from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import oauthlib.oauth1
import requests
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
consumer_key = os.getenv("TWITTER_CONSUMER_KEY")
consumer_secret = os.getenv("TWITTER_CONSUMER_SECRET")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
# Set the API endpoint and hashtag to search for
endpoint = 'https://api.twitter.com/1.1/search/tweets.json'
hashtag = 'ipl'

# Set the OAuth 1.0a client
client = oauthlib.oauth1.Client(consumer_key, client_secret=consumer_secret, resource_owner_key=access_token, resource_owner_secret=access_token_secret)

# Set the search query parameters
query_params = {
    'q': f'%23{hashtag}',
    'result_type': 'recent',
    'count': '5'
}

# Construct the URL with the query parameters
url = f"{endpoint}?{'&'.join([f'{k}={v}' for k,v in query_params.items()])}"

# Sign the request with OAuth 1.0a credentials
url, headers, body = client.sign(url, http_method='GET')

# Send the GET request to the API endpoint
response = requests.get(url, headers=headers)

# Create a dictionary to store the tweet texts and IDs
tweets_dict = {}

# Extract the tweet texts and IDs and store them in the dictionary
for tweet in response.json()['statuses']:
    tweets_dict[tweet['id_str']] = tweet['text'] + "\n"

# Print the dictionary of tweet texts and IDs
print(tweets_dict)

# Load the model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

# Perform sentiment analysis for each tweet
for tweet_id, tweet_text in tweets_dict.items():
    tweet_words = []
    
    for word in tweet_text.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    
    tweet_proc = " ".join(tweet_words)

    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
    output = model(**encoded_tweet)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    sentiment_results = []
    for i in range(len(scores)):
        l = labels[i]
        s = scores[i]
        sentiment_results.append(f"{l}: {s:.2f}")

    print(f"Tweet ID: {tweet_id}")
    print(f"Tweet Text: {tweet_text}")
    print(f"Sentiment Results: {' | '.join(sentiment_results)}\n")
