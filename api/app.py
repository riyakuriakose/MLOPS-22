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
    
    

