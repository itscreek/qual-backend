from rest_framework import serializers
from .models import GameLog

class GameLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = GameLog
        fields = ['event_type', 'event_time', 'word_to_type', 'key_pressed', 'correct']
