from src.pipelines.prediction_pipeline import CustomData, Prediction

from flask import Flask, request, render_template, jsonify
from flask import request


application = Flask(__name__)

app = application

@app.route('/')
def home():
    return render_template("page.html")

@app.route('/price_predictor',methods=['POST'])
def predict_operation():
    if (request.method == 'POST'):
        
        data = CustomData(
            carat=request.form['carat'],
            depth=request.form['depth'],
            table=request.form['table'],
            x=request.form['x'],
            y=request.form['y'],
            z=request.form['z'],
            cut=request.form['cut'],
            color=request.form['color'],
            clarity=request.form['clarity']
        )
        new_data = data.get_data_as_dataframe()
        
        # Prediction Pipeline;
        predict_pipleline = Prediction()
        predict_result=predict_pipleline.predict(new_data)
        predict_result = round(predict_result[0],2)
        return render_template('page.html',result=predict_result)
    
if  __name__ == '__main__':
        # app.run(host="0.0.0.0")
        app.run(host="0.0.0.0", port=5000, debug=True)
