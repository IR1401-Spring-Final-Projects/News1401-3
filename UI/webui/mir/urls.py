from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='mir-home'),
    path('classify/', views.cluster, name='mir-classify'),
    path('cluster/', views.classify, name='mir-cluster'),
]