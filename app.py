from flask import Flask, render_template, request
import pickle
import numpy as np

goldmodel = pickle.load(open('GoldPriceDecisionTree.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/gold_price")
def gold_price():
    return render_template('gold-price.html')






@app.route('/predict_gold', methods=['POST'])
def predict_gold_price():
    SPX = float(request.form.get('SPX'))
    USO = float(request.form.get('USO'))
    SLV = float(request.form.get('SLV'))
    EUR_USD = float(request.form.get('EUR_USD'))

    #gold price prediction
    goldresult = goldmodel.predict(np.array([SPX, USO, SLV, EUR_USD]).reshape(1,4))

    return render_template('gold-price.html', goldresult=goldresult)

if __name__ == '__main__':
    app.run(debug=True)