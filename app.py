from flask import Flask, render_template, request
import pickle
import numpy as np

sv = pickle.load(open('weather.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    
    data1 = request.form['temp_max']
    data2 = request.form['temp_min']
    data3 = request.form['wind']
    tot_data = [[data1,data2,data3]]
    arr = np.array(tot_data,dtype=float)
    pred = sv.predict(arr)
    weather = pred[0]
    html_content = f"<html><head></head><body style='background-color:grey'><center><br><br><h1> MACHINE LEARNING PREDICTION</h1><br><br><h1> LOGISTIC REGRESSION </h1><br><br><h1> The weather is  {weather} </h1></center></body></html>"
    with open("templates\prediction.html",'w') as html_file:
        html_file.write(html_content)
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
