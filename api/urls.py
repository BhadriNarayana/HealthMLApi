from django.contrib import admin
from django.urls import path, include
from . import views
urlpatterns = [
    path('', views.home, name = 'home'),
    path('test/', views.api, name = 'api'),
     path('test2/', views.api2, name = 'api2')
]
