from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import sklearn
from flask import jsonify
from datetime import date

housedata = pd.read_csv('Datasets/house_loc_dataframe.csv')

goldmodel = pickle.load(open('GoldPriceDecisionTree.pkl', 'rb'))
loanmodel = pickle.load(open('Loan_Status_RandomForest.pkl', 'rb'))
carmodel = pickle.load(open('Car_price_randomforest_regression.pkl', 'rb'))
laptopmodel = pickle.load(open('Laptop_RandomForest_Regressor.pkl', 'rb'))
housemodel = pickle.load(open('Banglore_house_LinearRegression.pkl', 'rb'))


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
    locations = sorted(housedata['location'].unique())
    return render_template('house-price.html', locations=locations)




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

@app.route('/predict_car', methods=['POST'])
def predict_car_price():

    year = int(request.form.get('year'))
    Present_Price = float(request.form.get('Present_Price'))
    Kms_Driven = int(request.form.get('Kms_Driven'))
    Owners = int(request.form.get('Owners'))

    Fuel_Type = request.form.get('Fuel_Type')
    Fuel_Type_Petrol = 0
    Fuel_Type_Diesel = 0
    if (Fuel_Type == 'Petrol'):
        Fuel_Type_Petrol = 1
        Fuel_Type_Diesel = 0
    elif (Fuel_Type == 'Diesel'):
        Fuel_Type_Petrol = 0
        Fuel_Type_Diesel = 1
    else:
        Fuel_Type_Petrol = 0
        Fuel_Type_Diesel = 0

    current_date = date.today()
    year = current_date.year - year

    Seller_Type = request.form.get('Seller_Type')
    Seller_Type_Individual =0
    if (Seller_Type == 'Individual'):
        Seller_Type_Individual = 1
    else:
        Seller_Type_Individual = 0

    Transmission_type = request.form.get('Transmission_type')
    Transmission_Mannual = 0
    if (Transmission_type == 'Mannual'):
        Transmission_Mannual = 1
    else:
        Transmission_Mannual = 0

    # Car Price Prediction
    prediction = carmodel.predict([[Present_Price, Kms_Driven, Owners, year, Fuel_Type_Diesel, Fuel_Type_Petrol,
                                 Seller_Type_Individual, Transmission_Mannual]])
    output = round(prediction[0], 2)
    if output < 0:
        return render_template('car-price.html', carresult="Sorry you cannot sell this car")
    else:
        return render_template('car-price.html', carresult="You Can Sell The Car at {} lakh ".format(output))


@app.route('/predict_Laptop', methods=['POST'])
def predict_laptop_price():

    Comp = request.form.get('Comp')
    Type = request.form.get('Type')
    ram = int(request.form.get('ram'))
    weight = float(request.form.get('weight'))
    touch = request.form.get('touch')
    if (touch == 'Yes'):
        touch = 1
    else:
        touch = 0

    ips = request.form.get('weight')
    if (ips == 'Yes'):
        ips = 1
    else:
        ips = 0

    scr_sz = float(request.form.get('scr_sz'))
    scr_res = request.form.get('scr_res')
    CPU = request.form.get('CPU')
    HDD = int(request.form.get('HDD'))
    SSD = int(request.form.get('SSD'))
    GPU = request.form.get('GPU')
    OS = request.form.get('OS')

    # Coverting to PPI
    X_res = int(scr_res.split('x')[0])
    Y_res = int(scr_res.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / scr_sz

    # Laptop Price Prediction
    prediction = laptopmodel.predict(np.array([Comp, Type, ram, weight, touch, ips,
                                 ppi, CPU, HDD, SSD, GPU, OS]).reshape(1,12))

    output = round(prediction[0])
    if output < 0:
        return render_template('laptop-price.html', Laptopresult="Sorry insert valid value.")
    else:
        return render_template('laptop-price.html', Laptopresult="You Can Buy This Configuration Laptop at Approx. {}/- ".format(output))


@app.route('/predict_house', methods=['POST'])
def predict_house_price():
    loc = request.form.get('loc')
    area = request.form.get('area')
    BHK = request.form.get('BHK')
    Bath = request.form.get('Bath')

    # house price prediction

    housevalue = pd.DataFrame([[loc, area, Bath, BHK]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    houseresult = housemodel.predict(housevalue)[0] * 1e5

    houseoutput = round(houseresult)
    if houseoutput < 0:
        return render_template('house-price.html', housetext="Sorry insert valid value.")
    else:
        return render_template('house-price.html', housetext="You Can Buy This House at Approx. {}/- ".format(houseoutput))




if __name__ == '__main__':
    app.run(debug=True)