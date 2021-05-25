import tkinter as tk
import random
from tkinter import messagebox


window = tk.Tk()  #Makes main window
window.wm_title("T.M.S")
window.columnconfigure(0, {'minsize': 1020})
window.columnconfigure(1, {'minsize': 335})


frame=tk.Frame(window)
frame.grid(row=0,column=0,rowspan=5,sticky='N',pady=10)

frame2=tk.Frame(window)
frame2.grid(row=0,column=1)

frame3=tk.Frame(window)
frame3.grid(row=1,column=1)

frame4=tk.Frame(window)
frame4.grid(row=2,column=1)

frame5=tk.Frame(window)
frame5.grid(row=3,column=1)

frame2.rowconfigure(1, {'minsize': 250})
frame3.rowconfigure(1, {'minsize': 80})
frame4.rowconfigure(1, {'minsize': 150})
frame5.rowconfigure(1, {'minsize': 80})


lbl1 = tk.Label(frame,text='Vehicle Detection And Tracking',font = "verdana 12 bold")
lbl1.pack(side='top')

lbl2 = tk.Label(frame2,text='Vehicle Breaking Traffic Rule',font = "verdana 10 bold")
lbl2.grid(row=0,column=0,sticky ='S',pady=10)

lbl3 = tk.Label(frame3,text='Veicle Speed',font = "verdana 10 bold")
lbl3.grid(row=0,column=0,sticky ='S',pady=10)


lbl4 = tk.Label(frame4,text='Detected License Plate',font = "verdana 10 bold")
lbl4.grid(row=0,column=0)

lbl5 = tk.Label(frame5,text='Extracted License Plate Number',font = "verdana 10 bold")
lbl5.grid(row=0,column=0)

display1 = tk.Label(frame)
display1.pack(side='bottom')  #Display 1

display2 = tk.Label(frame2)
display2.grid(row=1,column=0) #Display 2


display3 = tk.Label(frame3,text="",font = "verdana 14 bold",fg='red')
display3.grid(row=1,column=0)

display4 = tk.Label(frame4)
display4.grid(row=1,column=0)

display5 = tk.Label(frame5,text="",font = "verdana 24 bold",fg='green')
display5.grid(row=1,column=0)

window.mainloop()