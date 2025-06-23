import os
import joblib
import smtplib
import pandas as pd
from django.shortcuts import render
from .forms import MessageForm
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# === Load pre-trained model and vectorizer ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VEC_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

vectorizer = joblib.load(VEC_PATH)
model = joblib.load(MODEL_PATH)


# === Email sending helper ===
def send_email(subject, body, to_email, from_email, from_password, smtp_server='smtp.gmail.com', smtp_port=465):
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(from_email, from_password)
        server.send_message(msg)
        server.quit()
        print("✅ Email sent successfully")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


# === Spam detection ===
def predict_message(message, to_email, from_email, from_password):
    try:
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)
        result = 'Spam' if prediction[0] == 1 else 'Not a spam'

        if result == 'Not a spam':
            subject = "Notification: Not a Spam Message"
            body = f"The following message was classified as not spam:\n\n{message}"
            send_email(subject, body, to_email, from_email, from_password)

        return result
    except Exception as e:
        return f"Prediction failed: {e}"


# === View ===
def Home(request):
    result = None
    if request.method == 'POST':
        form = MessageForm(request.POST)
        if form.is_valid():
            message = form.cleaned_data['text']
            result = predict_message(
                message,
                'olasunkami314@gmail.com',  # To
                'nbail8931@gmail.com',      # From
                'pbsmluuzutmkrbwi'          # App password
            )
    else:
        form = MessageForm()

    return render(request, 'home.html', {'form': form, 'result': result})
