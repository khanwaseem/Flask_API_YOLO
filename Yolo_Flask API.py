from flask import jsonify,Flask,request
app = Flask(__name__)
import base64
import cv2
#import xml.etree.ElementTree as ET
import numpy as np
scale = 0.003
conf_threshold = 0.1
nms_threshold = 0.1
label = None

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, w, h):
    global frame
    global label
    global axle_detected
    color = COLORS[class_id]
    label = str(classes[class_id])
    return label


with open("obj.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
@app.route('/postjson', methods = ['POST'])
def postJsonHandler():
    global label
    label = "White"
    content = request.get_json()
    imgdata = base64.b64decode(content["name"])
    nparr = np.fromstring(imgdata, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    Width = frame.shape[1]
    Height = frame.shape[0]
    blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(w), round(h))
    
    return jsonify({"PlateColor":label})        


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080,debug=True)







