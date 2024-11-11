# Libraries / Modules
import random
import time
import speech_recognition as sr
import pyttsx3  
from transformers import pipeline
from datasets import load_dataset  
import torch
import soundfile as sf

recognizer = sr.Recognizer()

engine = pyttsx3.init()
engine.setProperty('rate', 150)

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id) 

chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")

# Loading datasets 
maths_data = load_dataset('HuggingFace-DataSet/operation_MATHS')  
writing_data = load_dataset('HuggingFace-DataSet/random_Words')  
pronunciation_data = load_dataset('HuggingFace-DataSet/random_Words')  
listening_data = load_dataset('HuggingFace-DataSet/random_Sentence') 

def listen_to_child():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source) 
        audio = recognizer.listen(source)
        
    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio) 
        print(f"Child said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return None
    except sr.RequestError:
        print("Speech recognition service is unavailable.")
        return None

# speaking AI's response (Text-to-Speech)
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Chatbot response generator based on user input
def chatbot_interaction(user_input):
    response = chatbot(user_input)
    return response[0]['generated_text']


def generate_math_problem(level):
    problem_data = maths_data['train']  
    if level == "easy":
        problem = random.choice([item for item in problem_data if item['difficulty'] == 'easy'])
    elif level == "medium":
        problem = random.choice([item for item in problem_data if item['difficulty'] == 'medium'])
    else:
        problem = random.choice([item for item in problem_data if item['difficulty'] == 'hard'])
    
    question = problem['question']
    answer = problem['answer']
    return question, answer

# Functions to generate prompt
def generate_writing_prompt():
    writing_words = writing_data['train']
    word = random.choice(writing_words)['word']
    prompt = f"Please write a sentence using the word '{word}'."
    return prompt, word

def generate_pronunciation_prompt():
    pronunciation_words = pronunciation_data['train']
    word = random.choice(pronunciation_words)['word']
    return word

def generate_listening_prompt():
    listening_sentences = listening_data['train']
    sentence = random.choice(listening_sentences)['sentence']
    return sentence

def adjust_difficulty(score):
    if score >= 8:
        return "hard"
    elif score >= 5:
        return "medium"
    else:
        return "easy"

def give_feedback(is_correct, activity_type):
    if activity_type == "math":
        if is_correct:
            speak("Great job! Keep it up!")
        else:
            speak("Oops, that's not quite right. Let's try another one.")
    elif activity_type == "writing":
        if is_correct:
            speak("Well done! Your sentence is perfect.")
        else:
            speak("It seems like you made a mistake. Let's try again.")
    elif activity_type == "pronunciation":
        if is_correct:
            speak("Great pronunciation!")
        else:
            speak("That was close. Try saying the word again.")
    elif activity_type == "listening":
        if is_correct:
            speak("Correct! Well done.")
        else:
            speak("That’s not quite right. Try again.")

# Main function 
def main():
    speak("Hello! I'm your AI tutor. Let's start learning.")
    level = "easy"  

    while True:
        speak("What would you like to practice today? Maths, Writing, Pronunciation, or Listening?")
        user_input = listen_to_child()
        
        if user_input:
            if "maths" in user_input.lower():
                # Math 
                problem, answer = generate_math_problem(level)
                speak(problem)
                child_answer = listen_to_child()
                if child_answer and int(child_answer) == answer:
                    speak("Correct!")
                    score = 10
                else:
                    speak(f"Oops, the correct answer was {answer}.")
                    score = 0
                
                level = adjust_difficulty(score)

            elif "writing" in user_input.lower():
                # Writing 
                prompt, word = generate_writing_prompt()
                speak(prompt)
                child_input = listen_to_child()
                speak(f"You wrote: {child_input}")
                
                if word in child_input.lower():
                    speak("Great sentence!")
                    score = 10
                else:
                    speak("Try again, use the word in your sentence.")
                    score = 0
                
                level = adjust_difficulty(score)
            
            elif "pronunciation" in user_input.lower():
                # Pronunciation 
                word = generate_pronunciation_prompt()
                speak(f"Please say the word '{word}'.")
                child_input = listen_to_child()
                if word.lower() in child_input.lower():
                    speak("Great pronunciation!")
                    score = 10
                else:
                    speak("Try again, say the word correctly.")
                    score = 0
                
                level = adjust_difficulty(score)
            
            elif "listening" in user_input.lower():
                # Listening
                sentence = generate_listening_prompt()
                speak(f"Listen carefully: {sentence}")
                time.sleep(2)
                speak("What did you hear?")
                child_input = listen_to_child()
                if sentence.lower() in child_input.lower():
                    speak("Correct! Well done.")
                    score = 10
                else:
                    speak("That’s not quite right. Try again.")
                    score = 0
                
                level = adjust_difficulty(score)
            
            else:
                speak("Sorry, I didn’t understand that. Please choose a valid option.")

        time.sleep(1)
# run application
if __name__ == "__main__":
    main()
