# -*- coding:utf-8 -*-
import pathlib
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import ImageFont, Image, ImageDraw
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import matplotlib.pyplot as plt
import numpy as np
import cv2


class UI(ttk.Frame):

    def __init__(self, master):
        super().__init__(master, padding=15)
        self.pack(fill=BOTH, expand=YES)

        """application variables"""
        _path = pathlib.Path().absolute().as_posix()
        self.save_var = ttk.StringVar(value=_path)
        self.path_var = ttk.StringVar(value=_path)
        self.sig_var = ttk.StringVar(value='PYTHON')
        self.test_var = ttk.StringVar(value=_path)

        """header and labelframe option container"""
        option_text = "选择图片以添加水印"
        self.option_lf = ttk.Labelframe(self, text=option_text, padding=15)
        self.option_lf.grid(row=0, column=0, columnspan=4)

        test_text = "选择图片以检测水印"
        self.test_lf = ttk.LabelFrame(self, text=test_text, padding=15)
        self.test_lf.grid(row=1, column=0, columnspan=4)

        self.create_path_row()
        self.create_sig_row()
        self.create_test_row()

    """define the UI"""

    def create_path_row(self):
        """Add path row to labelframe"""
        path_row = ttk.Frame(self.option_lf)
        path_row.grid(row=0, column=0, sticky=W)

        path_lbl = ttk.Label(path_row, text="文件路径", width=8)
        path_lbl.grid(row=0, column=1, sticky=W)

        path_ent = ttk.Entry(path_row, textvariable=self.path_var, width=50)
        path_ent.grid(row=0, column=2, padx=5)

        browse_btn = ttk.Button(
            master=path_row,
            text="浏览",
            command=self.on_browse,
            width=8
        )
        browse_btn.grid(row=0, column=3, padx=10)

    def create_sig_row(self):
        """Add path row to labelframe"""
        content_row = ttk.Frame(self.option_lf)
        content_row.grid(row=1, column=0, sticky=W)

        content_lbl = ttk.Label(content_row, text="水印内容", width=8)
        content_lbl.grid(row=0, column=1, sticky=W)

        content_ent = ttk.Entry(
            content_row, textvariable=self.sig_var, width=50)
        content_ent.grid(row=0, column=2, padx=5)

    def create_test_row(self):
        """Add result treeview to labelframe"""
        path_row = ttk.Frame(self.test_lf)
        path_row.grid(row=0, column=0, sticky=W)

        path_lbl = ttk.Label(path_row, text="文件路径", width=8)
        path_lbl.grid(row=0, column=0, sticky=W)

        path_ent = ttk.Entry(path_row, textvariable=self.test_var, width=50)
        path_ent.grid(row=0, column=2, padx=5)

        browse_btn = ttk.Button(
            master=path_row,
            text="浏览",
            command=self.on_browse_test,
            width=8
        )
        browse_btn.grid(row=0, column=3, padx=10)

    """define the functions"""

    def testsig(self, path):
        test = cv2.imread(path)
        test = test[:, :, 0]
        plt.imshow(np.log(1+np.abs(np.fft.fftshift(np.fft.fft2(test)))), 'gray')
        plt.show()

    def createsig(self, str):
        height = 540
        width = 960
        color = (0, 0, 0)
        img = Image.new('RGB', (width, height), color)
        fontpath = "font/simsun.ttc"
        font = ImageFont.truetype(fontpath, 256)

        draw = ImageDraw.Draw(img)
        w, h = font.getsize(str)
        # 绘制文字信息
        draw.text(((width-w)/2, (height-h)/2), str,
                  font=font, fill=(96, 96, 96))
        save = np.array(img)
        return save

    def show(self, img_path, sig_array):
        img = cv2.imread(img_path)

        B, G, R = cv2.split(img)
        img = cv2.merge((R, G, B))

        plt.figure(1, figsize=(10, 5))

        plt.subplot(3, 5, 6), plt.imshow(
            img), plt.title('picture'), plt.axis('off')

        height, width, channels = img.shape
        R_img_array = np.zeros(
            (height, width, 3), dtype=np.uint8)  # 初始化Red图片，全部值为0
        G_img_array = np.zeros(
            (height, width, 3), dtype=np.uint8)  # 初始化Green图片，全部值为0
        B_img_array = np.zeros(
            (height, width, 3), dtype=np.uint8)  # 初始化Blue 图片，全部值为0

        R_img_array[:, :, 0] = R
        G_img_array[:, :, 1] = G
        B_img_array[:, :, 2] = B

        sig = cv2.resize(sig_array, (int(0.5 * width), int(0.5 * height)),
                         interpolation=cv2.INTER_LINEAR)

        sig_add = np.zeros((height, width, 3), dtype=np.uint8)
        sig_add[int(height/2):height, int(width/2):width, :] = sig
        sig_add[0:int(height/2), 0:int(width/2), :] = sig[::-1, ::-1, :]

        plt.subplot(3, 5, 11), plt.imshow(
            sig_add), plt.title('signature'), plt.axis('off')

        plt.subplot(3, 5, 2), plt.imshow(
            R_img_array), plt.title('red'), plt.axis('off')
        plt.subplot(3, 5, 7), plt.imshow(
            G_img_array), plt.title('green'), plt.axis('off')
        plt.subplot(3, 5, 12), plt.imshow(
            B_img_array), plt.title('blue'), plt.axis('off')

        R_shift2center = np.fft.fftshift(np.fft.fft2(R))
        R_log_shift2center = np.log(1 + np.abs(R_shift2center))
        plt.subplot(3, 5, 3), plt.imshow(R_log_shift2center,
                                         'gray'), plt.title('log'), plt.axis('off')
        R_phase = np.angle(R_shift2center)
        plt.subplot(3, 5, 4), plt.imshow(
            R_phase, 'gray'), plt.title('phase'), plt.axis('off')

        G_shift2center = np.fft.fftshift(np.fft.fft2(G))
        G_log_shift2center = np.log(1 + np.abs(G_shift2center))
        plt.subplot(3, 5, 8), plt.imshow(G_log_shift2center,
                                         'gray'), plt.title('log'), plt.axis('off')
        G_phase = np.angle(G_shift2center)
        plt.subplot(3, 5, 9), plt.imshow(
            G_phase, 'gray'), plt.title('phase'), plt.axis('off')

        B_shift2center = np.fft.fftshift(np.fft.fft2(B))

        # 逆变换--两者合成
        B_amplitude = ((1+np.abs(B_shift2center))) * \
            np.exp(sig_add[:, :, 2]*0.02)-1  # 取振幅
        B_phase = np.angle(B_shift2center)
        B_real = B_amplitude*np.cos(B_phase)  # 取实部
        B_imag = B_amplitude*np.sin(B_phase)  # 取虚部
        B_reverse = np.zeros((height, width), dtype=complex)
        B_reverse.real = np.array(B_real)  # 重新赋值s1给s2
        B_reverse.imag = np.array(B_imag)
        B_back = np.abs(np.fft.ifft2(np.fft.ifftshift(B_reverse)))

        B_reverse_log = np.log(1+np.abs(np.fft.fftshift(np.fft.fft2(B_back))))

        plt.subplot(3, 5, 13), plt.imshow(B_reverse_log,
                                          'gray'), plt.title('log'), plt.axis('off')

        plt.subplot(3, 5, 14), plt.imshow(
            B_phase, 'gray'), plt.title('phase'), plt.axis('off')

        img_reverse = np.zeros((height, width, 3), dtype=np.uint8)

        R_reverse = np.abs(np.fft.ifft2(np.fft.ifftshift(R_shift2center)))
        img_reverse[:, :, 0] = R_reverse
        G_reverse = np.abs(np.fft.ifft2(np.fft.ifftshift(G_shift2center)))
        img_reverse[:, :, 1] = G_reverse
        img_reverse[:, :, 2] = B_back

        plt.subplot(3, 5, 10), plt.imshow(
            img_reverse), plt.title('reverse'), plt.axis('off')

        save_path = self.save_var.get()

        cv2.imwrite(save_path + '/save1.png', img_reverse[:, :, ::-1])

        plt.show()

    def on_browse(self):
        """Callback for directory browse"""
        path = askopenfilename(
            title="浏览文件", filetypes=[("photos", ("*.png", "*.jpg", "*.jpeg"),)])

        sig = self.sig_var.get()
        sig_array = self.createsig(sig)
        if path:
            self.path_var.set(path)
            self.show(path, sig_array)

    def on_browse_test(self):
        """Callback for browse test img"""
        path = askopenfilename(
            title="浏览文件", filetypes=[("photos", ("*.png", "*.jpg", "*.jpeg"),)]
        )

        if path:
            self.test_var.set(path)
            self.testsig(path)


if __name__ == '__main__':
    root = ttk.Window('盲水印', resizable=(0, 0))
    UI(root)
    root.mainloop()
