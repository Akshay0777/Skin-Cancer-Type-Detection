from tkinter import *
from tkinter.filedialog import askopenfilename
# from PIL import ImageTk
from tkinter import messagebox
import cv2
import numpy as np
import os
from multiprocessing import Process
import webbrowser
global filename


## Prediction functions
def predictionfun():
    filename = ""
    predbtn.place_forget()
    adminbtn.place_forget()
    predframe.place(height=500,width=1000,x=150,y=120)

    def open_file():
        file = askopenfilename(initialdir = "",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
        if file == "":
            messagebox.showerror("File is not input","Please enter File Name")
        elif file is not None:
            # print(file)
            filename = file
            # bar = Progressbar(predframe, length=180, style='grey.Horizontal.TProgressbar')
            # bar['value'] = 50
            # bar.pack(side = TOP, pady = 150)
            btn1 = Button(predframe, text ='find result',bg="skyblue", command = lambda:predictfun(filename))
            btn1.place(x=450,y=297)
            return filename
        else:
            messagebox.showerror("File is not input","Please enter File Name")

    def open_file1():
        filename = open_file()
        print(filename)

    def predictfun(filename):
        from keras.models import load_model
        import cv2
        model = load_model('model_keras.h5')
        image = str(filename)
        size = 32
        I = cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (size,size), interpolation=cv2.INTER_CUBIC)
        I = np.array(I)
        I = I.reshape(1,size,size,3)
        pred = model.predict(I)
        types = ['Melanocytic nevi (nv)' , 'Melanoma (mel)' , 'Benign keratosis-like lesions (bkl)' , 'Basal cell carcinoma (bcc)' , 'Actinic keratoses (akiec)' , 'Vascular lesions (vas)' , 'Dermatofibroma (df)']
        id = np.argmax(pred)
        result = types[id]
        print(f"You Have {result} type of Skin cancer")
        resultframe.place(height=500,width=1000,x=150,y=120)
        resultn = Label(resultframe,text="You have : ",fg="red", font = ('Helvetica',20))
        resultn.place(x=80,y=100)
        resultnb = Label(resultframe,text=result,fg="red",bg="white", font = ('Helvetica',40),width=100)
        resultnb.place(x=100,y=200)
        resultn = Label(resultframe,text=result,fg="red", font = ('Helvetica',40))
        resultn.place(x=100,y=200)
        

    openl = Label(predframe,text="Select your File :- ")
    openl.place(x=350,y=100)
    btnopen = Button(predframe, text ='Select', command = lambda:open_file1())
    btnopen.place(x=510,y=97)


## Admin Login
def loginf():
    username = luname1.get()
    password = lpassw1.get()
    print(username,password)
    if username == "":
        messagebox.showerror("Error","Username Required *")
        luname1.focus_set()
    elif password == "":
        messagebox.showerror("Error","Password Required *")
        lpassw1.focus_set()
    else:
        un = "admin"
        up = "admin"
        # mydb = mysql.connector.connect(host = "localhost" ,user = "root" ,password = "12345" ,database = "login")
        # cur = mydb.cursor()
        # cur.execute("select * from loginuser where username = %s && password = %s ",(username,password))
        # myresult = cur.fetchone()
        if un != username:
            messagebox.showerror("Error","enter valid username")
            print("enter valid username")
        elif up != password:
            messagebox.showerror("Error","enter valid password")
            print("enter valid password")
        else:
            print("yes present")
            messagebox.showinfo("successful","Login Successful")
            training()

def login():
    predbtn.place_forget()
    adminbtn.place_forget()
    loginframe.place(height=500,width=1000,x=150,y=120)
    

## Training Models
def training():
    predbtn.place_forget()
    adminbtn.place_forget()
    trainframe.place(height=500,width=1000,x=150,y=120)
    trainbtn.place(x=200,y=100)

def trainmodel():
    print("Training model in process")
    waittxt = Label(trainframe,text="Please wait for a Few Minutes",bg="white", font = ('Helvetica',20))
    waittxt.place(x=350,y=200)
    waittxt2 = Label(trainframe,text="Please do not close it, it will run in background automatically",bg="white", font = ('Helvetica',10))
    waittxt2.place(x=350,y=300)
    trainbtn.place_forget()
    
    p1 = Process(target=trainfunction)
    p1.start()


def trainfunction():
    print("Training")
    os.system('train.py')
    donetxt = Label(trainframe,text="Training is Done",bg="white", font = ('Helvetica',40))
    donetxt.place(x=350,y=200)
    print("Training is done")





## Back Buttons

def backpredf():
    resultframe.place_forget()
    print("work")

def backadmin():
    pass

def backhome():
    trainframe.place_forget()
    predframe.place_forget()
    loginframe.place_forget()
    predbtn.place(x=400,y=200)
    adminbtn.place(x=800,y=200)
    # linkl.place(x=400,y=475)
    # link.place(x=500,y=475)


## Link
def callback(url):
    webbrowser.open_new_tab(url)

if __name__ == '__main__':
    root = Tk()
    root.geometry("1300x700+25+20")
    root.title("Skin Cancer Detection")
    bg = PhotoImage(file = "images/bg.png")
    
    # Show image using label
    label1 = Label( root, image = bg)
    label1.place(x = 0, y = 0,relwidth=1,relheight=1)
    


    predframe = Frame(root,bg="skyblue")
    loginframe = Frame(root,bg="white")
    trainframe= Frame(root,bg="orange")
    resultframe= Frame(root,bg="white")

    weltxt = Label(root,text="Skin Cancer Detection",bg="white", font = ('Helvetica',20))
    weltxt.pack()

    predbtn = Button(root,text="Detection",bg="skyblue",command=predictionfun, height=5,width=30)
    predbtn.place(x=300,y=200)

    adminbtn = Button(root,text="Admin",bg="skyblue",command=login, height=5,width=30)
    adminbtn.place(x=830,y=200)

    weltxt1 = Label(loginframe,text="Login Admin", font = ('Helvetica',16))
    weltxt1.place(x=430,y=80)
    luname = Label(loginframe,text="Username",bg="white")
    luname.place(x=400,y=150)
    luname1 = Entry(loginframe)
    luname1.place(x=400,y=200,width=200)
    lpassw = Label(loginframe,text="Password",bg="white")
    lpassw.place(x=400,y=250)
    lpassw1 = Entry(loginframe,show = '*')
    lpassw1.place(x=400,y=300,width=200)
    lremember = Checkbutton(loginframe,text="remenber me",bg="white")
    lremember.place(x=420,y=350)
    lsubmit = Button(loginframe,text="Login",bg="skyblue",command= loginf)
    lsubmit.place(x=470,y=400)

    trainbtn = Button(trainframe,text="Train Model",bg="orange",command=trainmodel)
    

    backlogin = Button(loginframe,text="back",command=backhome)
    backlogin.place(x=10,y=10)
    backtrain = Button(trainframe,text="back",command=backhome)
    backtrain.place(x=10,y=10)
    backpred = Button(predframe,text="back",command=backhome)
    backpred.place(x=10,y=10)
    backtopred = Button(resultframe,text="back",command=backpredf)
    backtopred.place(x=10,y=10)

    waittxt = Label(trainframe,text="Please wait for a Few Minutes",bg="white", font = ('Helvetica',20))
    waittxt2 = Label(trainframe,text="Please do not close it, it will run in background automatically",bg="white", font = ('Helvetica',10))

    

    footer = Label(root,text="all Copyrights reserved",fg="white",bg="black")
    footer.place(x=0,y=675 ,width=1300)

    root.mainloop()