from django.contrib import admin

# Register your models here.

from apple.models import *

admin.site.site_header = 'Apple Dashboard Admin Panel'

admin.site.register(Tweet)
admin.site.register(TwitterUser)