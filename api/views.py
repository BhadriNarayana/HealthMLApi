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

    return render(request, 'api/home.html')




import json
@api_view(['GET'])
def api(request):
    
    if request.method == 'GET':
        print(request.body)
        
        image_str = request.data.get('image')

        # Decode the base64-encoded image data into bytes

        img_b = image_str.encode('utf-8')

        
        decoded_image = base64.b64decode(img_b)

        image_data = io.BytesIO(decoded_image)

        


        img = Image.open(image_data)
        #img.save('glioma.png')
    
        img = np.array(img, dtype=np.float64)


        img = tf.image.resize(img, (256, 256))
        img = tf.expand_dims(img, 0)

        pred = model.predict(img)

        label = labels[tf.argmax(tf.squeeze(pred))]

        return Response({'label': label})
        


