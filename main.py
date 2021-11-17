import os
from flask import Flask, send_from_directory, request, jsonify
from model import DecisionTreeClassifier, KNNClassifier, NeuralNetClassifier
import numpy as np

app = Flask(__name__, static_folder='ui')


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


@app.route('/', methods=['POST'])
def send_result():
    classifier = request.json['classifier']
    sdrr = float(request.json['sdrr'])
    medianRR = float(request.json['medianRR'])
    hr = float(request.json['hr'])
    lf = float(request.json['lf'])
    rmssd = float(request.json['rmssd'])
    pnn50 = float(request.json['pnn50'])
    hf = float(request.json['hf'])
    hflf = float(request.json['hflf'])
    meanRR = float(request.json['meanRR'])
    vec = np.array([[sdrr, medianRR, hr, lf, rmssd, pnn50, hf, hflf, meanRR]])
    vec.reshape(1, -1)

    if classifier == 'Decision Tree':
        res = DecisionTreeClassifier(vec)
    elif classifier == 'Weighted k-NNC':
        res = KNNClassifier(vec)
    else:
        res = NeuralNetClassifier(vec)
    return jsonify({"result": res}), 200


if __name__ == '__main__':
    app.run(use_reloader=True, port=5000)
