import pyttsx3 # It is a text to speech module in python

#it makes the ai speak through pyttsx3 module which we imported
def speak(Text):
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    engine.setProperty('voices', voices[1].id)
    engine.setProperty('rate', 170)
    print("   ")
    print(f" Monk: {Text}")
    engine.say(text=Text)
    engine.runAndWait()
    print("  ")


