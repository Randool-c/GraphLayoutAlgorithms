import os
import json

from os.path import join as pjoin
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

from frontend import path

# Create your views here.


def index(request):
    datasetnames = [x for x in os.listdir(path.DATASET_DIR) if os.path.isdir(pjoin(path.DATASET_DIR, x))]
    datasetnames = sorted(datasetnames)
    return render(request, 'index.html', context={
        'datasetnames': datasetnames,
    })


@require_http_methods(['POST'])
@csrf_exempt
def request_dataset_info(request):
    rsp = {'success': False, 'data': None}
    print('caonima')
    try:
        print('hello world')
        print(request.POST)
        datasetname = request.POST['datasetname']
        print('datasetname: ', datasetname)
        if os.path.isfile(pjoin(path.DATASET_DIR, datasetname, 'metadata.json')):
            with open(pjoin(path.DATASET_DIR, datasetname, 'metadata.json'), 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                f.close()

            rsp['success'] = True
            rsp['data'] = metadata
            return JsonResponse(rsp, status=200)
        else:
            return JsonResponse(rsp, status=404)
    except FileNotFoundError:
        return JsonResponse(rsp, status=404)
