from flask import Flask,send_file,jsonify,request
# from trained import traindf
from models import load
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/api/upload",methods=['POST'])
def index():
    uploaded_file = request.files['file']
    file_data = load.predictdata(uploaded_file)
    return jsonify(file_data)

@app.route("/api/v1",methods=['GET'])
def hello():
    return "Hello World"


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)

# if __name__ == '__main__':
#     load.predictdata()