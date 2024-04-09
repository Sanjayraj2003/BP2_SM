from flask import Flask, render_template, request

import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

p_s = PorterStemmer() 

def stemming(content):
    con = re.sub('[^a-z,A-z]',' ', content)
    con = con.lower()
    con = con.split()
    con = [p_s.stem(word) for word in con if not word in stopwords.words('english')]
    con = ' '.join(con)
    return con

app = Flask(__name__)

vector = pickle.load(open('vector.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get the news article from the form submission
        news_input = request.form.get("newsInput")

        # Preprocess the input
        news_input = stemming(news_input)
        inp_data = [news_input]

        # Vectorize the input
        vf = vector.transform(inp_data)

        # Make prediction
        pred = model.predict(vf)

        print(pred)
        
        # Return prediction result
        if pred:
            return "<div class='fake'>This news article is likely fake.</div>"
        else:
            return "<div class='real'>This news article is likely real.</div>"


if __name__ == "__main__":
    app.run(debug=True)
