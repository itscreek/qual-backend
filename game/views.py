from django.http import JsonResponse
from game.models import TypingWord

from game.domain.prediction_model import QuantumTypeTimePredictionModel

def problems(request):
    DEFAULT_NUM_RESPONSE_WORDS = 15
    num_words_for_response = DEFAULT_NUM_RESPONSE_WORDS
    num_words_in_db = TypingWord.objects.all().count()
    if num_words_in_db < num_words_for_response:
        # TODO: サーバーが準備完了時には適当な文字列が入っていて欲しいが、今はこれで対応
        import random, string
        sample_words = [''.join(random.choice(string.ascii_lowercase) for _ in range(10)) for _ in range(100)]
        for word in sample_words:
            TypingWord(word=word).save()
    
    # Chose random words from the database
    words_for_response = TypingWord.objects.order_by('?')[:num_words_for_response]
    response = JsonResponse({"words": [word.word for word in words_for_response]})
    return response
