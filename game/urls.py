from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import GameLogViewSet

router = DefaultRouter()
router.register('game', GameLogViewSet)

urlpatterns = [
    path('', include(router.urls)),
]
