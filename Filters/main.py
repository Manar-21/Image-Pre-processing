import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2

class Filters:

    def IntensitySlicing(self):
        try:
            self.root.destroy()
            IntensitySlicing = self.img.copy()
            for r in range(0, self.row):
                for c in range(0, self.col):
                    if self.img[r][c] > 150 and self.img[r][c] < 255:
                        IntensitySlicing[r][c] = 200
                    else:
                        IntensitySlicing[r][c] = 50
            self.showImage(IntensitySlicing)
        except:
            pass

    def ContrastStretching(self):
        try:
            self.root.destroy()
            ContrastStretching = self.img.copy()
            q_bitImg = 8
            MP = np.power(2, q_bitImg) - 1
            a = np.min(self.img.copy())
            b = np.max(self.img.copy())
            R = b - a
            for i in range(0, self.row):
                for j in range(0, self.col):
                    r = self.img[i][j]
                    s = (r / R) * MP
                    ContrastStretching[i][j] = round(s)
            self.showImage(ContrastStretching)
        except:
            pass

    def PowerLawTransformations(self):
        try:
            self.root.destroy()
            PowerLawTransformations = self.img.copy()
            c = 2
            lm = 0.9
            for i in range(0, self.row):
                for j in range(0, self.col):
                    r = self.img[i][j]
                    s = c * (np.power(r, lm))
                    PowerLawTransformations[i][j] = s
            self.showImage(PowerLawTransformations)
        except:
            pass

    def logTransformation(self):
        try:
            self.root.destroy()

            log_Transformation = self.img.copy()
            c = 20
            for i in range(0, self.row):
                for j in range(0, self.col):
                    r = self.img[i][j]
                    s = c * np.log(1 + r)
                    log_Transformation[i][j] = s
            self.showImage(log_Transformation)
        except:
            pass

    def NegativeTransformations(self):
        try:
            self.root.destroy()
            NegativeTransformations = self.img.copy()
            L = 256
            for i in range(0, self.row):
                for j in range(0, self.col):
                    r = self.img[i][j]
                    s = L - 1 - r
                    NegativeTransformations[i][j] = s
            self.showImage(NegativeTransformations)
        except:
            pass

    def HistogramEqualization(self):
        try:
            self.root.destroy()
            HistogramEqualization = cv2.equalizeHist(self.img)
            self.showImage(HistogramEqualization)
        except:
            pass

    def Max(self):
        try:
            self.root.destroy()
            Max = self.img.copy()
            for r in range(1, self.row-1):
                for c in range(1, self.col-1):
                    arr = np.array(
                        [self.img[r - 1][c + 1], self.img[r][c + 1], self.img[r + 1][c + 1], self.img[r - 1][c], self.img[r][c], self.img[r + 1][c],
                         self.img[r - 1][c - 1], self.img[r][c - 1], self.img[r + 1][c - 1]])
                    s = np.sort(arr)
                    Max[r][c] = s[-1]
            self.showImage(Max)
        except:
            pass

    def Min(self):
        try:
            self.root.destroy()
            Min = self.img.copy()
            for r in range(1, self.row-1):
                for c in range(1, self.col-1):
                    arr = np.array(
                        [self.img[r - 1][c + 1], self.img[r][c + 1], self.img[r + 1][c + 1], self.img[r - 1][c], self.img[r][c], self.img[r + 1][c],
                         self.img[r - 1][c - 1], self.img[r][c - 1], self.img[r + 1][c - 1]])
                    s = np.sort(arr)
                    s = np.sort(arr)
                    Min[r][c] = s[1]
            self.showImage(Min)
        except:
            pass

    def Median(self):
        try:
            self.root.destroy()
            Median = self.img.copy()
            for r in range(1, self.row-1):
                for c in range(1, self.col-1):
                    arr = np.array(
                        [self.img[r - 1][c + 1], self.img[r][c + 1], self.img[r + 1][c + 1], self.img[r - 1][c], self.img[r][c], self.img[r + 1][c],
                         self.img[r - 1][c - 1], self.img[r][c - 1], self.img[r + 1][c - 1]])
                    s = np.sort(arr)
                    Median[r][c] = s[5]
            self.showImage(Median)
        except:
            pass

    def Mean(self):
        try:
            self.root.destroy()
            Mean = self.img.copy()
            for r in range(0, self.row-1):
                for c in range(0, self.col-1):
                    arr = np.array(
                        [self.img[r - 1][c + 1], self.img[r][c + 1], self.img[r + 1][c + 1], self.img[r - 1][c], self.img[r][c], self.img[r + 1][c],
                         self.img[r - 1][c - 1], self.img[r][c - 1], self.img[r + 1][c - 1]])
                    s = round(np.sum(arr)/9)
                    Mean[r][c] = s
            self.showImage(Mean)
        except:
            pass

    def Gaussian(self):
        try:
            self.root.destroy()
            global sum
            Gaussian = self.img.copy()
            gauss = (1.0 / 57) * np.array(
                [[0, 1, 2, 1, 0],
                 [1, 3, 5, 3, 1],
                 [2, 5, 9, 5, 2],
                 [1, 3, 5, 3, 1],
                 [0, 1, 2, 1, 0]])

            sum(sum(gauss))

            for i in range(2, self.row - 2):
                for j in range(2, self.col - 2):
                    sum = 0
                    for k in range(-2, 3):
                        for l in range(-2, 3):
                            a = self.img.item(i + k, j + l)
                            p = gauss[2 + k, 2 + l]
                            sum = sum + (p * a)
                    b = sum
                    Gaussian.itemset((i, j), b)
            self.showImage(Gaussian)
        except:
            pass

    def Laplace(self):
        self.root.destroy()
        ddepth = cv2.CV_16S
        Laplace = cv2.Laplacian(self.img, ddepth, ksize=3)
        self.showImage(Laplace)

    def Sobel(self):
        try:
            self.root.destroy()
            Sobel = self.img.copy()
            for i in range(1, self.row - 1):
                for j in range(1, self.col - 1):
                    gx = (self.img[i - 1][j - 1] + 2 * self.img[i][j - 1] + self.img[i + 1][j - 1]) - (
                            self.img[i - 1][j + 1] + 2 * self.img[i][j + 1] + self.img[i + 1][j + 1])
                    gy = (self.img[i - 1][j - 1] + 2 * self.img[i - 1][j] + self.img[i - 1][j + 1]) - (
                            self.img[i + 1][j - 1] + 2 * self.img[i + 1][j] + self.img[i + 1][j + 1])
                    Sobel[i][j] = min(255, np.sqrt(gx ** 2 + gy ** 2))
            self.showImage(Sobel)
        except:
            pass

    def Prewitt(self):
        try:
            self.root.destroy()
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
            kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)

            x = cv2.filter2D(self.img, cv2.CV_16S, kernelx)
            y = cv2.filter2D(self.img, cv2.CV_16S, kernely)

            # Turn uint8, image fusion
            absX = cv2.convertScaleAbs(x)
            absY = cv2.convertScaleAbs(y)
            Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
            self.showImage(Prewitt)
        except:
            pass

    def Robert(self):
        try:
            self.root.destroy()
            kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
            kernely = np.array([[0, -1], [1, 0]], dtype=int)

            x = cv2.filter2D(self.img, cv2.CV_16S, kernelx)
            y = cv2.filter2D(self.img, cv2.CV_16S, kernely)

            absX = cv2.convertScaleAbs(x)
            absY = cv2.convertScaleAbs(y)
            Robert = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
            self.showImage(Robert)
        except:
            pass

    def Unsharp_mask(self):
        try:
            self.root.destroy()
            kernel_size = (3, 3)
            sigma = 1.0
            amount = 1.0
            threshold = 0
            blurred = cv2.GaussianBlur(self.img, kernel_size, sigma)
            unsharp = float(amount + 1) * self.img - float(amount) * blurred
            unsharp = np.maximum(unsharp, np.zeros(unsharp.shape))
            unsharp = np.minimum(unsharp, 255 * np.ones(unsharp.shape))
            unsharp = unsharp.round().astype(np.uint8)
            if threshold > 0:
                low_contrast_mask = np.absolute(self.img - blurred) < threshold
                np.copyto(unsharp, self.img, where=low_contrast_mask)
            self.showImage(unsharp)
        except:
            pass

    def showImage(self, img):
        self.root = tk.Toplevel()
        self.root.title('Image')
        self.root.geometry('400x400+900+130')
        ico = Image.open('back.jpg')
        photo = ImageTk.PhotoImage(ico)
        self.root.wm_iconphoto(False, photo)

        canvas = tk.Canvas(self.root, width=400, height=400)
        canvas.pack()
        my_img = ImageTk.PhotoImage(image=Image.fromarray(img))
        canvas.create_image(0, 0, anchor=tk.NW, image=my_img)

        self.root.mainloop()

    def Orignal_Image(self):
        try:
            self.root.destroy()
        except:
            pass
        try:
            self.img = cv2.imread(self.filename,0)
            self.size = 400
            self.img = cv2.resize(self.img, (self.size, self.size))
            global row, col
            self.row, self.col = self.img.shape
            self.showImage(self.img)
        except:
            pass

    def Browse(self):
        global filename
        self.filename = filedialog.askopenfilename()
        self.Orignal_Image()

    ############ GUI ############

    def __init__(self):
        window = tk.Tk()
        window.title('Filters')
        window.geometry("760x650+5+10")
        ico = Image.open('back.jpg')
        photo = ImageTk.PhotoImage(ico)
        window.wm_iconphoto(False, photo)
        bk = Image.open("back.jpg")
        bk = bk.resize((760, 650), Image.ANTIALIAS)
        bg = ImageTk.PhotoImage(bk)
        label1 = tk.Label(window, image=bg)
        label1.place(x=0, y=0)

        try:
            Labl1 = tk.Label(window, text="Image Enhancement", font=("Helvetice", 20), bg='#b3b3ff', width=20).place(x=240, y=10)

            Labl2 = tk.Label(window, text="Spatial Operates", font=("Helvetice", 17), bg='#b3b3ff', width=15).place(x=220, y=70)

            Labl4 = tk.Label(window, text="Point processing", font=("Helvetice", 17), bg='#b3b3ff', width=15).place(x=10, y=120)
            Labl5 = tk.Label(window, text="intensity transformation", font=("Helvetice", 17), bg='#b3b3ff', width=20).place(x=30, y=160)
            Btn1 = tk.Button(window, text="Negative Transformations", font=("Helvetice", 15), width=20, activebackground="#b3b3b3",
                                             command=self.NegativeTransformations).place(x=40, y=200)
            Btn2 = tk.Button(window, text="Log Transformations", font=("Helvetice", 15), width=20, activebackground="#b3b3b3",
                                  command=self.logTransformation).place(x=40, y=245)
            Btn4 = tk.Button(window, text="PowerLaw Transformations", font=("Helvetice", 15), width=22, activebackground="#b3b3b3",
                                  command=self.PowerLawTransformations).place(x=40, y=290)
            Btn3 = tk.Button(window, text="Contrast Stretching", font=("Helvetice", 15), width=20, activebackground="#b3b3b3",
                                  command=self.ContrastStretching).place(x=40, y=335)
            Btn4 = tk.Button(window, text="Intensity Slicing", font=("Helvetice", 15), width=20, activebackground="#b3b3b3",
                                  command=self.IntensitySlicing).place(x=40, y=380)

            Labl6 = tk.Label(window, text="Histogram processing", font=("Helvetice", 17), bg='#b3b3ff', activebackground="#b3b3b3",
                                  width=20).place(x=30, y=440)
            Btn5 = tk.Button(window, text="Histogram Equalization", font=("Helvetice", 15), width=20, activebackground="#b3b3b3",
                                  command=self.HistogramEqualization).place(x=40, y=480)

            Labl7 = tk.Label(window, text="Spatial Filtering", font=("Helvetice", 17), bg='#b3b3ff', width=15).place(x=430, y=120)
            Labl8 = tk.Label(window, text="Linear Filters", font=("Helvetice", 16), bg='#b3b3ff', width=13).place(x=340, y=160)

            Labl9 = tk.Label(window, text="Smoothing Filters", font=("Helvetice", 15), bg='#b3b3ff', width=15).place(x=360, y=200)
            Btn6 = tk.Button(window, text="Mean Filter", font=("Helvetice", 15), width=16, activebackground="#b3b3b3",
                                  command=self.Mean).place(x=370, y=235)
            Btn7 = tk.Button(window, text="Gaussian Filter", font=("Helvetice", 15), width=16, activebackground="#b3b3b3",
                                  command=self.Gaussian).place(x=370, y=280)
            Btn7 = tk.Button(window, text="Unsharp Filter", font=("Helvetice", 15), width=16, activebackground="#b3b3b3",
                                  command=self.Unsharp_mask).place(x=370, y=325)

            Labl10 = tk.Label(window, text="Edge Enhancing Filters", font=("Helvetice", 15), bg='#b3b3ff', width=22).place(x=360, y=375)
            Btn8 = tk.Button(window, text="Sobel Filter", font=("Helvetice", 15), width=16, activebackground="#b3b3b3",
                                  command=self.Sobel).place(x=370, y=410)
            Btn9 = tk.Button(window, text="Prewitt Filter", font=("Helvetice", 15), width=16, activebackground="#b3b3b3",
                                  command=self.Prewitt).place(x=370, y=455)
            Btn10 = tk.Button(window, text="Laplace Filter", font=("Helvetice", 15), width=16, activebackground="#b3b3b3",
                                  command=self.Laplace).place(x=370, y=500)
            Btn11 = tk.Button(window, text="Robert Filter", font=("Helvetice", 15), width=16, activebackground="#b3b3b3",
                                   command=self.Robert).place(x=370, y=545)

            Labl11 = tk.Label(window, text="Non-Linear Filters", font=("Helvetice", 16), bg='#b3b3ff', width=15).place(x=550, y=160)
            Btn11 = tk.Button(window, text="Min Filter", font=("Helvetice", 15), width=13, activebackground="#b3b3b3",
                                   command=self.Min).place(x=570, y=200)
            Btn11 = tk.Button(window, text="Max Filter", font=("Helvetice", 15), width=13, activebackground="#b3b3b3",
                                   command=self.Max).place(x=570, y=245)
            Btn12 = tk.Button(window, text="Median Filter", font=("Helvetice", 15), width=13, activebackground="#b3b3b3",
                                   command=self.Median).place(x=570, y=290)

            Btn14 = tk.Button(window, text="Browse", font=("Helvetice", 15), width=20, activebackground="#b3b3b3",
                                   command=self.Browse).place(x=130, y=600)
            Btn15 = tk.Button(window, text="Original Image", font=("Helvetice", 15), width=20, activebackground="#b3b3b3",
                                   command=self.Orignal_Image).place(x=430, y=600)
        except:
            pass
        window.mainloop()

if __name__ == '__main__':
    Filters()
