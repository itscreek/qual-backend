# Generated by Django 5.0.6 on 2024-05-08 08:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0003_alter_gamelog_correct_alter_gamelog_key_pressed_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='gamelog',
            name='correct',
            field=models.BooleanField(null=True),
        ),
    ]
