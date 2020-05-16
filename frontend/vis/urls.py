from django.urls import path

from . import views

app_name = 'vis'
urlpatterns = [
    path('', views.index, name='index'),
]
