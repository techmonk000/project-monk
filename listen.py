import speech_recognition as sr  # It recognises our voice


def listenvoice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening to your voice ...")
        r.pause_threshold = 2.2
        audio = r.listen(source, 0, 2)
    try:
        print("Recognizing your command...")
        query = r.recognize_google(audio, language="en-in")
        print(f"Your command was : {query}\n")

    except:
        print("Failed to recognize your command...")
        query = ""

    query = str(query)
    return query.lower()

listenvoice()

