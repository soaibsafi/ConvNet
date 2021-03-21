import numpy as np
from PIL import ImageTk, Image
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import tensorflow
from tensorflow.keras.models import load_model

model = load_model('../model/CatDogModel-1-10epochs.h5')

classes = {
    0: 'Classified as Cat.',
    1: 'Classified as Dog.'
}

top = tk.Tk()
top.geometry('1280x720')
top.title('Conv2D Cat-Dog Classification')
top.configure(bg='#c4efff')

label = Label(top, bg='#c4efff', font=('Courier', 20, 'bold'))
score_label = Label(top, bg='#c4efff', font=('Courier', 20, 'bold'))

sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((128, 128))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = image/255
    pred = model.predict_classes([image])[0]
    score = model.predict([image])[0]
    score = 'Score: '+str(score)
    sign = classes[pred]
    print(score)
    label.configure(fg='#011638', text=sign)
    score_label.configure(fg='#011638', text=score)

def show_classify_button(file_path):
    classify_btn = Button(top, text='Classify Image', command=lambda: classify(file_path), padx=10, pady=10)
    classify_btn.configure(bg='#2f8296', fg='black')
    classify_btn.place(relx=0.80, rely=0.90)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path) 
        uploaded.thumbnail(((top.winfo_width()/1.25), top.winfo_height()/1.25))
        image = ImageTk.PhotoImage(uploaded)
        print('Reached')
        sign_image.configure(image=image)
        sign_image.image = image
        label.configure(text='')
        score_label.configure(text='')
        
        #print(file_path)
        show_classify_button(file_path)
    except:
        pass

upload = Button(top, text='Upload', command=upload_image, padx=20, pady=10)
upload.configure(bg='#2f8296', fg='black')
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
score_label.pack(side=BOTTOM, expand=True)
heading= Label(top, text='Conv2D Cat-Dog Classification', pady=20, font=('Courier', 30, 'bold'))
heading.configure(bg='#c4efff', fg='#000000')
heading.pack()
top.mainloop()