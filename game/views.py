from django.http import JsonResponse

def problems(request):
    response = JsonResponse({"words": ["apple", "banana", "cherry", "date", "elderberry"]})
    return response
