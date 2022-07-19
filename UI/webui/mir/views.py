from shutil import ExecError
from django.shortcuts import redirect, render
from django.http import HttpResponse, HttpResponseRedirect
from django.db import connection
from django.urls import reverse
from django.core.exceptions import FieldError
from django.db.models import Field
from django.apps import apps

TABLES = [x.__name__ for x in apps.get_models() if not x.__name__.startswith('Auth') and not x.__name__.startswith('Django')][:-6]
REQUEST_QUERY_KEY = 'query'
REQUEST_K_KEY = 'k'
BUTTON_SEARCH_KEY = 'search'
BUTTON_CLASSIFY_KEY = 'classify'
BUTTON_CLUSTER_KEY = 'cluster'
BUTTON_CANCEL_KEY = 'cancel'


def home(request):
    context = {
        'tables': TABLES,
        'header': [],
        'data': [],
    }
    if request.method == 'POST':
        if BUTTON_SEARCH_KEY in request.POST:
            search(request, context)
        elif BUTTON_CLUSTER_KEY in request.POST:
            return redirect('mir-cluster')
        elif BUTTON_CLASSIFY_KEY in request.POST:
            return redirect('mir-cluster')
    return render(request, 'mir/home.html', context)




def cluster(request):
    if request.method == 'POST' and BUTTON_CANCEL_KEY in request.POST:
        return redirect('mir-home')
    return render(request, 'mir/cluster.html')

def classify(request):
    if request.method == 'POST' and BUTTON_CANCEL_KEY in request.POST:
        return redirect('mir-home')
    return render(request, 'mir/cluster.html')


def search(request, context):
    query = request.POST.get(REQUEST_QUERY_KEY, None)
    k = request.POST.get(REQUEST_K_KEY, None)
    
    context['header'] = ['Title', 'Intro', 'Text', 'Category']
    temp = []
    for i in range(int(k)):
        temp.append({'Title': 1, 'Intro' : 2, 'Text' : 3, 'Category' : 4})
    context['data'] = temp


def about(request):
    return render(request, 'mir/about.html')
