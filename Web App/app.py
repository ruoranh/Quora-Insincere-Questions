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
    cv = joblib.load('CountVectRaw.pkl')
    model = joblib.load('TokenNBModel.pkl')


    if request.method == 'POST':
        question = request.form.get('rawtext')
        data = [question]
        vect = cv.transform(data)
        result = model.predict_proba(vect)[0][1]
        return render_template('result.html', prediction = round(result,2), question = question)


if __name__ == '__main__':
    app.run(debug=True)
