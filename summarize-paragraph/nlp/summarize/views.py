from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def summarize(request):
    return render(request, 'summarize/index.html')