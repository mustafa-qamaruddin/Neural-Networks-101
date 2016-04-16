from Tkinter import *

class View(Frame):
    master = None
    padding = 10
    width = 640
    height = 480

    def setFrame(self):
        self.master.title("Neural Networks Project 2016")
        self.master.maxsize(self.width, self.height)
        self.master.geometry("%dx%d+%d+%d" % (self.width, self.height, self.padding, self.padding))
        self.master.resizable(FALSE,FALSE)
        self.master.rowconfigure(5, weight=1)
        self.master.columnconfigure(2, weight=1)

        self.center(self.master)
        Frame.__init__(self, self.master)
        return

    def center(self, toplevel):
        toplevel.update_idletasks()
        w = toplevel.winfo_screenwidth()
        h = toplevel.winfo_screenheight()
        size = tuple(int(_) for _ in toplevel.geometry().split('+')[0].split('x'))
        x = w / 2 - size[0] / 2
        y = h / 2 - size[1] / 2
        toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))

    def say_hi(self):
        print "hi there, everyone!"

    def createWidgets(self):
        Label(self.master, text="Train States of Nature").grid(row=0, column=0, padx=self.padding, pady=self.padding)

        Label(self.master, text="First:").grid(row=1, column=0)
        Label(self.master, text="Second:").grid(row=2, column=0)

        self.e1 = Entry(self.master)
        self.e2 = Entry(self.master)

        self.e1.grid(row=0, column=1)
        self.e2.grid(row=1, column=1)

        self.cb = Checkbutton(self.master, text="Hardcopy")
        self.cb.grid(row=2, columnspan=2, sticky=W)

    def __init__(self, master=None):
        if master is None:
            master = Tk()

        self.master = master
        self.setFrame()
        self.createWidgets()

    def __del__(self):
        self.master.destroy()
