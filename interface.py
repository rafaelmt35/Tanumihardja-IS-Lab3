from tkinter import *
import numpy as np
from PIL import ImageGrab
import network

def clearwidget():
    global cv
    cv.delete("all")

    l1.destroy()
    l2.destroy()

def predict():
    global l1, l2
    global cv, window
    global n
    widget = cv
    # Setting co-ordinates of canvas
    x = window.winfo_rootx() + widget.winfo_x() + 30.5
    y = window.winfo_rooty() + widget.winfo_y() + 143.5
    x1 = x + widget.winfo_width() + 445
    y1 = y + widget.winfo_height() + 395
 
    # Image is captured from canvas and is resized to (28 X 28) px
    img = ImageGrab.grab().crop((x, y, x1, y1)).resize((10, 10))
    ImageGrab.grab().crop((x,y,x1,y1)).resize((10, 10)).save("digit.png")
    # Converting rgb to grayscale image
    img = img.convert('L')

    # # Extracting pixel matrix of image and converting it to a vector of (1, 784)
    array_image = np.asarray(img)
    array_image = array_image/255

    # Prediction
    prediction_digit, reliability = n.predict(array_image)

    l1 = Label(window, text="Prediction Digit = " + str(prediction_digit), font=('Algerian', 20))
    l1.place(x=30,y=550)
    l2 = Label(window, text="Reliability = " + str(reliability) + " %", font=('Algerian', 20))
    l2.place(x=30,y=580)

lastx, lasty = None, None

# Activate canvas
def event_activation(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y 
 
# To draw on canvas
def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=30, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y

window = Tk()
window.title("Neural Network")

window.geometry("500x650")
window.resizable(False, False)

# Canvas
cv = Canvas(window, bg="black", height="400", width="450")
cv.place(x=20, y=80)
cv.bind('<Button-1>', event_activation)

# Result label
l1 = Label()
l2 = Label()

title = Label(window, text="Handwritten Digit Recoginition", font=('Algerian', 25), fg="red")
title.place(x=90, y=10)

b1 = Button(window,text="PREDICT",command= predict,width=20,height=2)
b1.place(x=20, y=490)
b2 = Button(window,text="CLEAR",command= clearwidget,width=20,height=2)
b2.place(x=260, y=490)

n = network.Neural_Network((10, 10), 0.02, 3, 20)
n.set_test_weights_bias()

window.mainloop()
