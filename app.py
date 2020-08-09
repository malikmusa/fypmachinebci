import numpy as np
from flask import Flask, jsonify ,render_template,request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

port = int(os.environ.get('PORT', 5000))


@app.route('/')
def home():
    return "App is running!"

@app.route('/check')
def check():
    a = request.args.get('alpha')
    b = request.args.get('beta')
    g = request.args.get('gamma')
    int_features = float(a),float(b),float(g)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    return jsonify(status=200,message='success',prediction_text='Emotions {}'.format(output))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=port, debug=True)