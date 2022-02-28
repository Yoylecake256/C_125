#A classification model.

from flask import Flask, jsonify, request
from data import get_prediction

#creating an app.

app = Flask(__name__)

# creating an additional route
@app.route("/")
def index():
    return"Welcome to the home page, Tht's our API"

@app.route("/predit-digit",methods=["POST"])

def predit_data():
    image= request.files.get("digit")
    prediction= get_prediction(image)
    return jsonify({
        "prediction": prediction
    }),200

if __name__=="__main__":
    app.run(debug = True,port=8080)
