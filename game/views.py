from rest_framework import viewsets
from .models import GameLog
from .serializers import GameLogSerializer

class GameLogViewSet(viewsets.ModelViewSet):

    queryset = GameLog.objects.all()
    serializer_class = GameLogSerializer
