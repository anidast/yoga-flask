from __future__ import print_function
from flask import Flask, request
from flask_cors import CORS, cross_origin

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils import download

import os

app = Flask(__name__)
CORS(app)

@app.route("/")
def view():
    return "<h1>Try /estimate to get the pose estimation</h1>"

@app.route("/estimate")
def estimate():
    ctx = mx.current_context()
    deserialized_net = gluon.nn.SymbolBlock.imports("sposenet-symbol.json", ['data'], "sposenet-0001.params", ctx=ctx) 

    detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
    # Note that we can reset the classes of the detector to only include
    # human, so that the NMS process is faster.

    detector.reset_class(["person"], reuse_weights=['person'])

    im_fname = download('https://images.pexels.com/photos/617000/pexels-photo-617000.jpeg', path='image.jpg')
    # im_fname = download('http://i.huffpost.com/gen/1616446/images/o-YOGA-HEALTH-BENEFITS-facebook.jpg', path='image.jpg')

    x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
    print('Shape of pre-processed image:', x.shape)

    class_IDs, scores, bounding_boxs = detector(x)

    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)
    predicted_heatmap = deserialized_net(pose_input)
    pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
    prediction = pred_coords[0].asnumpy().tolist()

    return {'prediction' : prediction}

@app.route("/predict", methods = ['POST'])
@cross_origin()
def predict():
    ctx = mx.current_context()
    deserialized_net = gluon.nn.SymbolBlock.imports("sposenet-symbol.json", ['data'], "sposenet.params", ctx=ctx) 

    detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)

    detector.reset_class(["person"], reuse_weights=['person'])

    image = request.files['file']
    image.save(image.filename)

    x, img = data.transforms.presets.ssd.load_test(image.filename, short=512)
    print('Shape of pre-processed image:', x.shape)

    class_IDs, scores, bounding_boxs = detector(x)

    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)
    predicted_heatmap = deserialized_net(pose_input)
    pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
    prediction = pred_coords[0].asnumpy().tolist()

    return {'prediction' : prediction}


if __name__ == '__main__':
    app.run(debug='true' )