from tkinter import *
from tkinter import filedialog
from tkinter import messagebox

class View(Frame):
    master = None
    padding = 10
    width = 640
    height = 480
    ctrl = None

    def setFrame(self):
        self.master.title("Neural Networks Project 2016")
        self.master.maxsize(self.width, self.height)
        self.master.geometry("%dx%d+%d+%d" % (self.width, self.height, self.padding, self.padding))
        self.master.resizable(FALSE, FALSE)
        self.master.rowconfigure(10, weight=1)
        self.master.columnconfigure(3, weight=1)

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

    def handleFileInputWidget(self, obj_entry, obj_entry_label):
        dir_name = filedialog.askdirectory(parent=self.master)
        obj_entry.insert(END, dir_name)
        self.ctrl.importImagesFrmFolder(dir_name, obj_entry_label.get())
        messagebox.showinfo('Import Done', 'Images were loaded in Memory!')
        return

    def handleSIFT(self):
        self.ctrl.findKeyPoints()
        messagebox.showinfo('SIFT Done', 'SIFT for images in Memory was Calculated!')
        return

    def createWidgets(self):
        Label(self.master, text="Train States of Nature").grid(row=0, column=0, columnspan=3, padx=self.padding,
                                                               pady=self.padding, sticky=E)

        ## Labels
        self.l1 = Label(self.master, text="Images Folder[Class I]:")
        self.s1 = Label(self.master, text="Label[Class I]:")
        self.l2 = Label(self.master, text="Images Folder[Class II]:")
        self.s2 = Label(self.master, text="Label[Class I]:")
        self.l3 = Label(self.master, text="Images Folder[Class III]:")
        self.s3 = Label(self.master, text="Label[Class I]:")
        self.l4 = Label(self.master, text="Images Folder[Class IV]:")
        self.s4 = Label(self.master, text="Label[Class I]:")
        self.l5 = Label(self.master, text="Images Folder[Class V]:")
        self.s5 = Label(self.master, text="Label[Class I]:")

        ## Labels Grid
        self.l1.grid(row=1, column=0, sticky=W)
        self.s1.grid(row=1, column=2, sticky=W)
        self.l2.grid(row=2, column=0, sticky=W)
        self.s2.grid(row=2, column=2, sticky=W)
        self.l3.grid(row=3, column=0, sticky=W)
        self.s3.grid(row=3, column=2, sticky=W)
        self.l4.grid(row=4, column=0, sticky=W)
        self.s4.grid(row=4, column=2, sticky=W)
        self.l5.grid(row=5, column=0, sticky=W)
        self.s5.grid(row=5, column=2, sticky=W)

        ## Inputs
        self.e1 = Entry(self.master)
        self.e2 = Entry(self.master)
        self.e3 = Entry(self.master)
        self.e4 = Entry(self.master)
        self.e5 = Entry(self.master)

        ## Inputs Grid
        self.e1.grid(row=1, column=1, sticky=W + E)
        self.e2.grid(row=2, column=1, sticky=W + E)
        self.e3.grid(row=3, column=1, sticky=W + E)
        self.e4.grid(row=4, column=1, sticky=W + E)
        self.e5.grid(row=5, column=1, sticky=W + E)

        ## Inputs for Class Labels
        self.ee1 = Entry(self.master)
        self.ee2 = Entry(self.master)
        self.ee3 = Entry(self.master)
        self.ee4 = Entry(self.master)
        self.ee5 = Entry(self.master)

        ## Inputs Grid for Class Labels
        self.ee1.grid(row=1, column=3, sticky=W + E)
        self.ee2.grid(row=2, column=3, sticky=W + E)
        self.ee3.grid(row=3, column=3, sticky=W + E)
        self.ee4.grid(row=4, column=3, sticky=W + E)
        self.ee5.grid(row=5, column=3, sticky=W + E)

        ## Default Values for Class Labels
        self.ee1.insert(END, "Apple")
        self.ee2.insert(END, "Helicopter")
        self.ee3.insert(END, "Aeroplane")
        self.ee4.insert(END, "Cat")
        self.ee5.insert(END, "Laptop")

        ## Buttons
        self.b1 = Button(self.master, text="Open Class I", command=lambda: self.handleFileInputWidget(self.e1, self.ee1))
        self.b2 = Button(self.master, text="Open Class II", command=lambda: self.handleFileInputWidget(self.e2, self.ee2))
        self.b3 = Button(self.master, text="Open Class III", command=lambda: self.handleFileInputWidget(self.e3, self.ee3))
        self.b4 = Button(self.master, text="Open Class IV", command=lambda: self.handleFileInputWidget(self.e4, self.ee4))
        self.b5 = Button(self.master, text="Open Class V", command=lambda: self.handleFileInputWidget(self.e5, self.ee5))
        self.b6 = Button(self.master, text="Apply SIFT", command=self.handleSIFT)

        ## Buttons Grid
        self.b1.grid(row=1, column=4, sticky=W + E)
        self.b2.grid(row=2, column=4, sticky=W + E)
        self.b3.grid(row=3, column=4, sticky=W + E)
        self.b4.grid(row=4, column=4, sticky=W + E)
        self.b5.grid(row=5, column=4, sticky=W + E)
        self.b6.grid(row=6, columnspan=5, sticky=W + E)

    def __init__(self, ctrl, master=None):
        if master is None:
            master = Tk()

        self.ctrl = ctrl
        self.master = master
        self.setFrame()
        self.createWidgets()

    def __del__(self):
        self.master.destroy()
