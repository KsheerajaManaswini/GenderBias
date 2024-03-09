import nltk
import wikipedia

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, render_template, jsonify

nltk.download('punkt')
app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-gender-bias")
model = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-gender-bias")

def detect_gender_bias(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()
    score = logits.softmax(dim=1)[0][predicted_class].item()
    label = "True" if predicted_class == 1 else "False"
    result = "yes" if label == "True" else "no"
    return result, score

def split_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

def fetch_wikipedia_content(keyword):
    try:
        page = wikipedia.page(keyword)
        text = page.content
        return text
    except wikipedia.exceptions.PageError:
        return "Page not found."
    except wikipedia.exceptions.DisambiguationError:
        return "Multiple pages found. Please provide a more specific keyword."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch-content', methods=['POST'])
def fetch_content():
    keyword = request.form['keyword']
    text = fetch_wikipedia_content(keyword)

    # Split the content into sentences, ignoring the last sentence
    sentences = split_into_sentences(text)[:-1]

    # List to store gender-biased sentences
    gender_biased_sentences = []

    # List to store non-gender-biased sentences
    non_gender_biased_sentences = []

    # Iterate through sentences and check for gender bias
    for sentence in sentences:
        try:
            result, _ = detect_gender_bias(sentence)
            if result == "yes":
                gender_biased_sentences.append(sentence)
            else:
                non_gender_biased_sentences.append(sentence)
        except Exception as e:
            print(f"Error processing sentence: {str(e)}")
            continue

    # Create a summary paragraph of non-gender-biased sentences
    summary_paragraph = " ".join(non_gender_biased_sentences)
    
    return jsonify({'gender_biased_sentences': gender_biased_sentences, 'summary_paragraph': summary_paragraph})

if __name__ == '__main__':
    app.run(debug=True)
