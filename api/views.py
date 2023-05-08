from django.shortcuts import render
import tensorflow as tf
# Create your views here.
from django.templatetags.static import static

def home(request):
    url = static('api/brain_tumor.h5')
    model  = tf.keras.models.load_model(url)


    return render(request, 'api/home.html', {})

