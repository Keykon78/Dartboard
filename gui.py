import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab

from settings import Settings


class GUI:

    ENV = {
        'GUI_RESOLUTION_SCALE' : 0.5,
        'SHOW_GUI' : ["FEED","DIFFERENCE", "ARROW","DARTBOARD", "APEX", "ORIENTATION"]#['ORIENTATION','FEED',"ARROW1","ARROW2" ,"DIFFERENCE", "ROTATED", "DARTBOARD"]
    }

    @staticmethod
    def show(frame, window='feed'):
        if window in GUI.ENV['SHOW_GUI']:
            cv2.imshow(window, cv2.resize(frame, (int(frame.shape[1] * GUI.ENV['GUI_RESOLUTION_SCALE']), int(frame.shape[0] * GUI.ENV['GUI_RESOLUTION_SCALE']))))

    @staticmethod
    def imShow_debug(frame):
        if Settings.get_run_mode == 'debug':
            if(frame.ndim == 3):
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(frame,cmap='Greys_r')
            plt.show()
        else:
            return

    @staticmethod
    def imShow(frame):
        if (frame.ndim == 3):
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(frame, cmap='Greys_r')
        plt.show()

    @staticmethod
    def im_compare(imgs):


        if len(imgs) > 4:
            n_col = 4
        else:
            n_col = 2
        n_row = int(len(imgs)/n_col)
        _, axs = plt.subplots(n_row, n_col)
        axs = axs.flatten()

        for img, ax in zip(imgs, axs):
            if (img.ndim == 3):
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img, cmap='Greys_r')
        plt.show()


