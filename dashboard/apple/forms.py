from django import forms

class replyForm(forms.Form):
	reply = forms.CharField(label = 'reply', max_length = 280, widget=forms.TextInput(attrs={'name': 'reply', 'placeholder': 'Your reply', 'class': 'form-control' }))


from django.contrib.auth.forms import AuthenticationForm

# If you don't do this you cannot use Bootstrap CSS
class LoginForm(AuthenticationForm):
    username = forms.CharField(label="Username", max_length=30,
                               widget=forms.TextInput(attrs={'name': 'username', 'placeholder': 'Username'}))
    password = forms.CharField(label="Password", max_length=30,
                               widget=forms.PasswordInput(attrs={'name': 'password', 'placeholder': 'Password'}))

