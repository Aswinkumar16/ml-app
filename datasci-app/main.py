from flask import Flask,send_file,jsonify,request
# from trained import traindf
from models import load
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/upload",methods=['POST'])
def index():
    uploaded_file = request.files['file']
    file_data = load.predictdata(uploaded_file)
    return jsonify(file_data)


if __name__ == '__main__':
    app.run(port=8080)

# if __name__ == '__main__':
#     load.predictdata()