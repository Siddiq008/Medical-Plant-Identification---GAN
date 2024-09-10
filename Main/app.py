import json

from flask import Flask, render_template, request

import prediction

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/compute', methods=['POST'])
def compute():
    print("inside request")
    filename = request.form['image']
    print(filename)
    data = prediction.compute(filename)
    print(data)
    with open('templates\\SasiJson.json', 'r') as file:
        temp = json.load(file)
    for i in temp:
        print(f'Image Name = {i["name"]}')
        if i['name'].lower() == data.lower():
            return render_template("output.html", data=i)
    else:
        return "Image not found"


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
