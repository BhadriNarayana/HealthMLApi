from django.shortcuts import render
import tensorflow as tf
from rest_framework.decorators import api_view
from django.templatetags.static import static
import cv2

model  = tf.keras.models.load_model('static/brain_tumor.h5')
def home(request):
    img = cv2.imread('static/test_glioma.jpg')
    
    img = tf.image.resize(img, (256, 256))

    
    res = model.predict(tf.expand_dims(img, 0))
    

    return render(request, 'api/home.html', {'res': res})

