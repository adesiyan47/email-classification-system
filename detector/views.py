from django.shortcuts import render

# Create your views here.
import pandas as pd
from django.shortcuts import render
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from .forms import MessageForm
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


dataset = pd.read_csv('emails.csv')

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataset['text'])
X_train, X_test, y_train, y_test = train_test_split(X, dataset['spam'], test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)


# def predict_message(message):
#     message_vector = vectorizer.transform([message])
#     prediction = model.predict(message_vector)
#     return 'Spam' if prediction[0] == 1 else 'Not a spam'
def send_email(subject, body, to_email, from_email, from_password, smtp_server='smtp.gmail.com', smtp_port=465):
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        # Use SMTP_SSL for port 465
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(from_email, from_password)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

def predict_message(message, to_email, from_email, from_password):
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)
    result = 'Spam' if prediction[0] == 1 else 'Not a spam'

    if result == 'Not a spam':
        subject = "Notification: Not a Spam Message"
        body = f"The following message was classified as not spam:\n\n{message}"
        send_email(subject, body, to_email, from_email, from_password)

    return result

def Home(request):
    result = None
    if request.method == 'POST':
        form = MessageForm(request.POST)
        if form.is_valid():
            message = form.cleaned_data['text']
            result = predict_message(message,'olasunkami314@gmail.com', 'nbail8931@gmail.com','pbsmluuzutmkrbwi')     
    else:
        form = MessageForm()
    return render(request, 'home.html', {'form': form, 'result': result})


