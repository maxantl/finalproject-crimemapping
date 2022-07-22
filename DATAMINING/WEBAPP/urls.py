from django.urls import path
from . import views
urlpatterns = [
    path('', views.req, name='Final Project'),
    path("change_type", views.change_type),
]