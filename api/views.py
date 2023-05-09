from django.shortcuts import render
import tensorflow as tf
from rest_framework.decorators import api_view
from PIL import Image
import cv2
import numpy as np
from rest_framework.response import Response
import base64
import io

labels = ['glioma', 'meninigioma', 'notumor', 'pituitary']
model  = tf.keras.models.load_model('static/brain_tumor.h5')
def home(request):

    img = cv2.imread('static/test_glioma.jpg')
    
    img = tf.image.resize(img, (256, 256))

    
    res = model.predict(tf.expand_dims(img, 0))
    res_index = tf.math.argmax(tf.squeeze(res))

    res = labels[res_index]


    return render(request, 'api/home.html', {'res': res})


@api_view(['GET'])
def api(request):
    #data = request.data

    #file = request.files

    print("request", request.FILES['media'])

    imgInMemoryUploaded = request.FILES['media']
    
    imgPILObject = Image.open(img)
    
    img = np.array(img, dtype=np.float64)


    img = tf.image.resize(img, (256, 256))
    img = tf.expand_dims(img, 0)

    pred = model.predict(img)

    label = labels[tf.argmax(tf.squeeze(pred))]

    return Response({'label': label})

import json
@api_view(['GET'])
def api2(request):
    
    if request.method == 'GET':
        print(request.body)
        
        image_str = request.data.get('image')

        # Decode the base64-encoded image data into bytes

        img_b = image_str.encode('utf-8')

        
        decoded_image = base64.b64decode(img_b)

        image_data = io.BytesIO(decoded_image)

        


        img = Image.open(image_data)
        img.save('glioma.png')
    
        img = np.array(img, dtype=np.float64)


        img = tf.image.resize(img, (256, 256))
        img = tf.expand_dims(img, 0)

        pred = model.predict(img)

        label = labels[tf.argmax(tf.squeeze(pred))]

        return Response({'label': label})
        


