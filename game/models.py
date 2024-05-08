from django.db import models

class GameLog(models.Model):
    event_type = models.CharField(max_length=100)
    event_time = models.DateTimeField()
    word_to_type = models.CharField(max_length=100, blank=True)
    key_pressed = models.CharField(max_length=1, blank=True)
    correct = models.BooleanField(null=True)

    def __str__(self):
        return self.event_type
