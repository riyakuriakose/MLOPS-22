from flask import Flask
from flask import request
from flask import jsonify

app= Flask(__name__)

@app.route("/hello_world")
def hello_world():
    return("HI")
    
@app.route("/sum",methods=['POST'])
def sum():
    print(request.json)
    x= request.json['x']
    y= request.json['y']
    z=x+y
  
    return jsonify({"sum":z})

    
model_path = "svm_gamma_0.08_C_0.004.joblib"
@app.route("/predict",methods=['POST'])
def  predict_digit():
	image = request.json["image"]
	model = load(model_path)
	predicted = model.predict([image])
	return {"y_presicted": int(predcited[0])}

    
    

