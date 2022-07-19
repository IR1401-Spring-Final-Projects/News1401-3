from django.db import models

# Create your models here.

class News(models.Model):
    title = models.TextField()
    intro = models.TextField()
    text = models.TextField()
    category = models.TextField()
    