from django.urls import path

from . import views

app_name = 'vis'
urlpatterns = [
    path('', views.index, name='index'),
    path('request_dataset', views.request_dataset_info, name='request_dataset_info'),
    # path('')
]
