
from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from titanic_prediction import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', views.index, name='Homepage'),
    path('index/pred', views.pred, name='multiply'),
]
