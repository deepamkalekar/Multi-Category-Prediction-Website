from flask import Flask, render_template, request
import pickle
import numpy as np

goldmodel = pickle.load(open('GoldPriceDecisionTree.pkl', 'rb'))
loanmodel = pickle.load(open('Loan_Status_RandomForest.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/gold_price")
def gold_price():
    return render_template('gold-price.html')

@app.route("/loan_status")
def loan_status():
    return render_template('loan-status.html')

@app.route("/car_price")
def car_price():
    return render_template('car-price.html')


@app.route("/laptop_price")
def laptop_price():
    return render_template('laptop-price.html')

@app.route("/home_price")
def home_price():
    return render_template('house-price.html')




@app.route('/predict_gold', methods=['POST'])
def predict_gold_price():
    SPX = float(request.form.get('SPX'))
    USO = float(request.form.get('USO'))
    SLV = float(request.form.get('SLV'))
    EUR_USD = float(request.form.get('EUR_USD'))

    #gold price prediction
    goldresult = goldmodel.predict(np.array([SPX, USO, SLV, EUR_USD]).reshape(1,4))

    return render_template('gold-price.html', goldresult=goldresult)

@app.route('/predict_loan', methods=['POST'])
def predict_loan_status():
    gend = request.form.get('gend')
    if(gend ==  'Male'):
        gend = 1
    else:
        gend = 0
    marid = request.form.get('marid')
    if(marid == 'Yes'):
        marid=1
    else:
        marid=0
    depnd = int(request.form.get('depnd'))
    edu = request.form.get('edu')
    if(edu == 'Graduate'):
        edu=1
    else:
        edu=0
    slfemp = request.form.get('slfemp')
    if(slfemp=='Yes'):
        slfemp=1
    else:
        slfemp=0
    ainc = int(request.form.get('ainc'))
    cinc = int(request.form.get('cinc'))
    LoanAmount = int(request.form.get('LoanAmount'))
    lat = int(request.form.get('lat'))
    pltarea = request.form.get('pltarea')
    if(pltarea=='Rural'):
        pltarea=0
    elif(pltarea=='Semiurban'):
        pltarea=1
    else:
        pltarea=2

    #Loan Status Prediction
    loanresult = loanmodel.predict(np.array([gend, marid, depnd, edu, slfemp, ainc, cinc, LoanAmount, lat, pltarea]).reshape(1,10))
    if loanresult[0] == 1:
        loanresult = 'Congratulations you are eligible for Loan.'
    else:
        loanresult = 'Sorry, you are not eligible for Loan'

    return render_template('loan-status.html', loanresult=loanresult)

if __name__ == '__main__':
    app.run(debug=True)