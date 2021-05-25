### Restful Api for Number Plate Recognition and OCR

from flask import Flask, request, jsonify
import base64
from plate_detection import Predict

inputFileName = "inputRecieved.jpg"
imagePath = "apiData/" + inputFileName

predict = Predict()

app = Flask(__name__)


def decodeImageIntoBase64(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(image):
    with open("apiData/Output.jpg", 'wb') as f:
        f.write(image)
        f.close()
    with open("apiData/Output.jpg", "rb") as f:
        return base64.b64encode(f.read())


@app.route('/')
def home():
    content = {
    "POST": 'http://127.0.0.1:5000/predict',
    'Request Params': {'image': '(Image in base64 encoded string)'},
    'Output': {
    'plate Number': " ['FF671980'] "
    }
    }
    sample = " SAMPLE INPUT OUTPUT FORMATS FOR POST REQUEST "
    return jsonify(API=sample, sample=content)


@app.route("/predict", methods=["POST"])
def getPrediction():
    inpImage = request.json['image']
    decodeImageIntoBase64(inpImage, imagePath)
    output = predict.predict_from_image(imagePath, api=True)
    # image = encodeImageIntoBase64(output["opImage"])
    text = output["License_Plate"]
    return jsonify(plateNumber=text)


if __name__ == "__main__":
    app.run(debug=True)