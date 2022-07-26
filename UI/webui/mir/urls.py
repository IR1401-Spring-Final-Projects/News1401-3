from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='mir-home'),
    path('classify/', views.classify, name='mir-classify'),
    path('cluster/', views.cluster, name='mir-cluster'),
]