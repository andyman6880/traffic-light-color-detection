The model detects traffic lights from an image and identifies its color as red, green, or yellow.

The model is YOLO11n fine-tuned on a specific labelled traffic light dataset. Install the dependencies from requirements.txt

The image you want to test can go into the examples folder as "examples/input.jpg"
and the annotated output would go to "examples/output.jpg"

There is a sample image loaded in already.

The annotated output would put a rectangle around the traffic light(s) and have a colour label and confidence score.

It may struggle with smaller traffic lights or ones that are more visually obstructed, such as at night or partially covered.

The general precision for predicting a traffic light colour is around 92.6%, and is able to find about 92.5% of the traffic lights that were present.

Dataset from https://universe.roboflow.com/iaagl/traffic-lights-rqds4/dataset/1

License:
CC BY 4.0