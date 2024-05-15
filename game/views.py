from django.http import JsonResponse
from game.models import TypingWord

def problems(request):
    DEFAULT_NUM_RESPONSE_WORDS = 15
    num_words_in_db = TypingWord.objects.all().count()
    num_words_for_response = DEFAULT_NUM_RESPONSE_WORDS

    if num_words_in_db < num_words_for_response:
        response = JsonResponse({"words": [word.word for word in TypingWord.objects.all()]})
        return response
    
    # Chose random words from the database
    words_for_response = TypingWord.objects.order_by('?')[:num_words_for_response]
    response = JsonResponse({"words": [word.word for word in words_for_response]})
    return response
