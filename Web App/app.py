from flask import Flask, render_template, url_for, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.externals import joblib
    model = joblib.load('TokenNBModel.pkl')
    cv = joblib.load('CountVectRaw.pkl')

    if request.method == 'POST':
        question = request.form.get('rawtext')
        data = [question]
        vect = cv.transform(data)
        result = model.predict_proba(vect)[0][1]
        return render_template('result.html', prediction = result, question = question)


if __name__ == '__main__':
    app.run(debug=True)
