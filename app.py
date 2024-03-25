import requests
from PIL import Image, ImageOps
from io import BytesIO
from flask import Flask, request, jsonify
from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys
import xml.etree.ElementTree as ET

app = Flask(__name__)

LS_URL = "http://localhost:8080"
LS_API_TOKEN = "0a1eb952a1fc99f77c8434c34ee82cea06e95674"

class YOLOv8Backend(LabelStudioMLBase):
    def __init__(self,**kwargs):
       # Call base class constructor
        super(YOLOv8Backend, self).__init__(**kwargs)
      
        self.labels = ['colony']
        # Load model
        self.model = YOLO("best.pt")
    def parse_xml_to_dict(self, xml_string):
        root = ET.fromstring(xml_string)
        parsed_dict = {}
        
        for child in root:
            
            if child.tag == 'Image':
                view_info = {}
                view_info['from_name'] = child.attrib['name']
                view_info['inputs'] = [{
                    'type': 'image',
                    'value': child.attrib['value']
                }]
                parsed_dict['items'] = view_info
            elif child.tag == 'RectangleLabels':
                if 'items' not in parsed_dict:
                    parsed_dict['items'] = {}
                parsed_dict['items']['type'] = child.tag
                parsed_dict['items']['to_name'] = [child.attrib['toName']]
                labels = [label.attrib['value'] for label in child.findall('Label')]
                parsed_dict['items']['labels'] = labels
        
        return parsed_dict
    
    def predict(self, data, **kwargs):
        """
        Returns the list of predictions based on input list of tasks for 1 image
        """
        tasks = data.get('tasks')
        task = tasks[0]
        label_config = data.get('label_config')
        parsed_config = self.parse_xml_to_dict(label_config)
        image_url = task['data']['image']
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(parsed_config, 'RectangleLabels', 'image')
        # Getting URL of the image
        full_url = LS_URL + image_url
        # Header to get request
        header = {
            "Authorization": "Token " + LS_API_TOKEN}
        # Getting URL and loading image
        pil = Image.open(BytesIO(requests.get(
            full_url, headers=header).content))
        image = ImageOps.exif_transpose(pil)
        # Height and width of image
        original_width = image.width
        original_height = image.height
        # Creating list for predictions and variable for scores
        predictions = []
        score = 0
        i = 0
        print(original_width)
        print(original_height)
        # Getting prediction using model
        results = self.model.predict(image)
        # Getting mask segments, boxes from model prediction
        for result in results:
            # for i, prediction in enumerate(result.boxes):
            #     xyxy = prediction.xyxy[0].tolist()
            #     predictions.append({
            #         "id": str(i),
            #         "from_name": self.from_name,
            #         "to_name": self.to_name,
            #         "type": "rectanglelabels",
            #         "score": prediction.conf.item(),
            #         "original_width": original_width,
            #         "original_height": original_height,
            #         "image_rotation": 0,
            #         "value": {
            #             "rotation": 0,
            #             "x": (original_height-xyxy[1]) / original_height * 100, 
            #             "y": (xyxy[0]) / original_width * 100,
            #             "width": (xyxy[2] - xyxy[0]) / original_height * 100,
            #             "height": (xyxy[3] - xyxy[1]) / original_width * 100,
            #             "rectanglelabels": [self.labels[int(prediction.cls.item())]]
            #         }
            #     })
            #     score += prediction.conf.item()
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                polygon_points = [(box.xyxyn[0][2].item())*100,(box.xyxyn[0][3].item())*100,(box.xyxyn[0][0].item()-box.xyxyn[0][2].item())*100,(box.xyxyn[0][1].item()-box.xyxyn[0][3].item())*100]
                # print(type(self.to_name))
                # Adding dict to prediction
                predictions.append({
                    "from_name" : 'label',
                    "to_name" : self.to_name,
                    "id": str(i),
                    "type": "rectanglelabels",
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_rotation": 0,
                    "value": {
                        "rotation": 0,          
                        "x": polygon_points[0], "y": polygon_points[1],
                        "width": polygon_points[2], "height": polygon_points[3],
                        "rectanglelabels": [self.labels[int(box.cls[0].item())]]
                        }})
                # Calculating score
                score += box.conf[0].item()
                
        print(10*"#", "Returned Prediction", 10*"#")
         
        # Dict with final dicts with predictions
        final_prediction = {"model_version": "INITIAL","results":[{
            "result": predictions,
            "score": score / (i + 1),
            "model_version": "Yolov8"
        }]}

       
        return final_prediction
       

backend = YOLOv8Backend()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    results = backend.predict(data =data)
    final_data = jsonify(results)
    return final_data
    

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'UP',
        'model_class': 'YOLOv8'
    })

@app.route('/setup', methods=['POST'])
def setup():
    return jsonify({'model_version': 'Yolov8'})

if __name__ == '__main__':
    app.run(debug=True,host='localhost',port='9090')
