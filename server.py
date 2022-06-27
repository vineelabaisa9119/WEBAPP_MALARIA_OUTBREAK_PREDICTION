from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("outbreak_detect.csv")

app = Flask(__name__)
# deserializing to read the file

model1 = pickle.load(open("model1.pkl", 'rb'))


@app.route('/')
def index():
    return render_template("index.html")  # we are able to render to webpage


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    features = [int(x) for x in request.form.values()]
    print(features)
    final = [np.array(features)]
    x = df.iloc[:, 0:4].values
    sst = StandardScaler().fit(x)
    output = model1.predict(sst.transform(final))
    print(output)

    if output[0] == 1:
        return render_template('index.html', pred=f'YES..!There is Chance for MALARIA OUTBREAK')
    else:
        return render_template('index.html', pred=f'NO..!There is NO Chance for MALARIA OUTBREAK')


if __name__ == '__main__':
    app.run(debug=True)

