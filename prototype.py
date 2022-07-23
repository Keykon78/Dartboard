from arrow_detector import Arrow_Detector
from calcboard import Calcboard
from dartboard import Dartboard
from dartboard_detector import Dartboard_Detector
from difference_detector import Difference_Detector
from gui import GUI
from image_tools import Image_Tools
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from settings import Settings


class Prototype:
    ENV = {
        'VIDEO_INPUT_PATH': "./images/video.mov",
        'FPS_LIMIT': 20,
        'FPS_SKIP': 40, #40
        'RESOLUTION_WIDTH': 1080,
        'RESOLUTION_HEIGHT': 1920,

        'DIFFERENCE_THRES': 2 * 10 ** 5,
        'DIFFERENCE_THRES_RESET': 5 * 10 ** 5,
        'SHAKING_THRES': 1 * 10 ** 5
    }

    def __init__(self):
        self.dartboard_detector = Dartboard_Detector()
        self.difference_detector = Difference_Detector()
        self.dartboard = Dartboard()
        self.arrow_detector = Arrow_Detector()

        self.reset()
        self.capture()

    def compute_base_frame(self, img):

        print("Computing Baseframe")
        # img = cv.imread('Photos/T1.png')

        self.BASE_IM = Image_Tools.white_balance(img)

        GUI.im_compare([img, self.BASE_IM ])
        g_mask = self.dartboard.get_green_mask(self.BASE_IM)
        r_mask = self.dartboard.get_red_mask(self.BASE_IM)
        self.x, self.y, self.w, self.h = Image_Tools.calc_roi(g_mask, r_mask)

        self.BASE_IM = Image_Tools.zoom_roi(self.BASE_IM, self.x, self.y, self.w, self.h)



        self.BASE_IM_GRAY = cv.cvtColor(self.BASE_IM.copy(), cv.COLOR_BGR2GRAY)



        self.M = self.dartboard.get_orientation(self.BASE_IM)
        self.M_corrected = self.dartboard.warp_board(self.BASE_IM_GRAY, self.M, 1000)
        self.M_corrected_c = self.dartboard.warp_board(self.BASE_IM, self.M, 1000)

        g_mask = self.dartboard.get_green_mask(self.M_corrected_c)
        r_mask = self.dartboard.get_red_mask(self.M_corrected_c)

        crop_black, crop_bg_black = self.dartboard.crop_new(self.M_corrected_c)
        self.white, white_in_cat = self.dartboard.get_white_contours(crop_bg_black)
        self.black, black_in_cat = self.dartboard.get_black_contours(crop_black)
        crop_rings, crop_bg_black = self.dartboard.crop_board(self.M_corrected_c, r_mask, g_mask, scale=0.95)

        r_crop_mask = self.dartboard.get_red_mask(crop_rings)


        crop_rings[350:600, 350:600] = (255, 255, 255)
        g_crop_mask = self.dartboard.get_green_mask(crop_rings)


        # GUI.imShow(crop_rings)
        self.red, hierarchy = cv.findContours(r_crop_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.green, hierarchy = cv.findContours(g_crop_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        self.red += Image_Tools.create_noise(r_crop_mask)
        self.green += Image_Tools.create_noise(g_crop_mask)

        self.calcboard = Calcboard(white_in_cat, black_in_cat, r_crop_mask, g_crop_mask, self.M_corrected_c,
                                   self.dartboard.warp_mode)

        # self.contour_roi = self.dartboard.draw_contours(self.M_corrected_c, self.white, self.black, self.red,
        # self.green)

        # GUI.imShow(self.contour_roi)

    def reset(self):
        print("-- SYSTEM RESET--")
        self.BASE_IM = None
        self.BASE_IM_GRAY = None
        self.BASE_IM_green = None
        self.BASE_IM_red = None
        self.BASE_IM_board = None
        self.ROI_x = None
        self.ROI_y = None
        self.ROI_w = None
        self.ROI_h = None
        self.M_corrected = None
        self.M = None
        self.PREVIOUS_DIFFERENCE = None

    def capture(self):
        # video = cv.VideoCapture(Settings.get_cam_input())
        video = cv.VideoCapture('../Tests/dart-videos/dart-video-08.mp4')
        skakeframe = 200
        frame_counter = 0
        shaking_thresh = -1
        shaker =[]
        prev_diff = 0
        shaking_img = None
        while True:
            ret, new_frame = video.read()
            if new_frame is None:
                print("END OF VIDEO, BREAKING")
                break
            if frame_counter % Prototype.ENV['FPS_SKIP'] == 0:
                if self.BASE_IM is None:
                    self.compute_base_frame(new_frame)

                    ref_img = self.M_corrected
                    GUI.imShow(ref_img)

                    continue

                IM_ROI = Image_Tools.zoom_roi(new_frame, self.x, self.y, self.w, self.h)
                # dst = cv.fastNlMeansDenoisingColored(IM_ROI, None, 10, 10, 7, 21)
                IM_ROI = Image_Tools.white_balance(IM_ROI)
                IM_ROI_grey = cv.cvtColor(IM_ROI, cv.COLOR_BGR2GRAY)

                warped_frame = self.dartboard.warp_board(IM_ROI_grey, self.M, 1000)
                IM_ROI_difference, IM_ROI_GRAY_NORM, IM_ROI_GRAY2_NORM, IM_ROI_GRAY_NORM_DIFF, arrow_diff = self.difference_detector.computeDifference(
                    ref_img, warped_frame)
                IM_ROI_difference_blur = cv.medianBlur(IM_ROI_difference, 11)
                difference_sum = np.sum(IM_ROI_difference_blur)
                #arrow_diff = cv.medianBlur(arrow_diff, 11) # Vielleicht rausnehmen
                # if shaking_thresh == -1 and frame_counter > skakeframe:
                #     print("shaking_find")
                #     shaking_thresh = np.mean(shaker)
                #     print("Shaking Img")
                #     shaking_img = cv.bitwise_not(shaking_img)
                #
                #     Prototype.ENV['FPS_SKIP'] = 40
                #     #GUI.imShow(IM_ROI_difference)
                #     # plt.plot(shaker)
                #     # plt.show()
                # elif shaking_thresh == -1:
                #     print("Shaking Calib: " + str(int(frame_counter/skakeframe*100)) + "%", end="\r")
                #     if difference_sum > prev_diff:
                #         prev_diff = difference_sum
                #         shaking_img = IM_ROI_difference
                #     shaker.append(difference_sum)
                #     frame_counter += 1
                #     continue
                # # if difference_sum > Prototype.ENV['DIFFERENCE_THRES_RESET']:
                # #     self.compute_base_frame(new_frame)
                # #     continue
                # if difference_sum < shaking_thresh:
                #     # self.reset()
                #
                #     continue
                if difference_sum > Prototype.ENV['DIFFERENCE_THRES']:
                    #print("Frame: ", frame_counter)
                    # print("Mode: ",self.dartboard.warp_mode)
                    #GUI.imShow(IM_ROI_difference)
                    #arrow_cleaned = cv.bitwise_and(shaking_img, arrow_diff)
                    #GUI.im_compare([arrow_diff, arrow_cleaned])

                    arrow_x1, arrow_y1 = self.arrow_detector.detectArrow(
                        arrow_diff, self.dartboard.warp_mode, warped_frame)


                    IM_ROI_grey = self.arrow_detector.markApex(warped_frame, arrow_x1, arrow_y1)
                    score = self.calcboard.calc_score(arrow_x1, arrow_y1)
                    print("Score: ", score)
                    #GUI.im_compare([arrow_diff, IM_ROI_grey])
                    # GUI.imShow(IM_ROI_difference)
                    # GUI.imShow(IM_ROI_grey)

                    ref_img = warped_frame
            frame_counter += 1


if __name__ == "__main__":
    prototype = Prototype()
