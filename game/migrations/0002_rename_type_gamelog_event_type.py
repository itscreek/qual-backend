# Generated by Django 5.0.6 on 2024-05-08 07:57

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='gamelog',
            old_name='type',
            new_name='event_type',
        ),
    ]
