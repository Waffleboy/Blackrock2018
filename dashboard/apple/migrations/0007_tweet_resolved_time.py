# -*- coding: utf-8 -*-
# Generated by Django 1.11.3 on 2018-04-07 08:18
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('apple', '0006_tweet_priority'),
    ]

    operations = [
        migrations.AddField(
            model_name='tweet',
            name='resolved_time',
            field=models.DateTimeField(blank=True, default=None, null=True),
        ),
    ]
