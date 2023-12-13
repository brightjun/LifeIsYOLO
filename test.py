from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import requests
from ultralytics import YOLO
"/Users/jun/Github/YOLO/img.jpeg"
ROOTDIR = "/home/ubuntu/LifeIsYOLO"
#ROOTDIR = "/home/ubuntu/LifeIsYOLO"

app = Flask(__name__)

def predict_objects_from_url(image_url):
    try:
        # Download the image from the URL
        response = requests.get(image_url)
        #respose code :200 이 아니면 오류.
        response.raise_for_status()

        # Load the image
        img = Image.open(BytesIO(response.content))

        # Initialize YOLO model
        model = YOLO("yolov8n.pt")

        # Use the model for prediction
        results = model.predict(img)

        # Convert results to JSON format
        result_json = results[0].tojson()
        img_size = results[0].orig_shape

        # Create a dictionary with image size and labels
        result_dict = {"imgSize": img_size, "Labels": result_json}

        return result_dict

    except Exception as e:
        return {"error": str(e)}


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "image_url" not in data:
        return jsonify({"error": "Missing 'image_url' parameter"}), 400

    image_url = data["image_url"]
    result = predict_objects_from_url(image_url)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
