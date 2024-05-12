from django.http import JsonResponse
from game.models import TypingWord

def problems(request):
    DEFAULT_NUM_RESPONSE_WORDS = 15
    num_words_in_database = TypingWord.objects.all().count()
    num_response_words = DEFAULT_NUM_RESPONSE_WORDS

    if num_words_in_database < num_response_words:
        response = JsonResponse({"words": [word.word for word in TypingWord.objects.all()]})
        return response
    
    # Chose random words from the database
    words_to_response = TypingWord.objects.order_by('?')[:num_response_words]
    response = JsonResponse({"words": [word.word for word in words_to_response]})
    return response
