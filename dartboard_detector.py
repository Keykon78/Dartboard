import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from gui import GUI
from image_tools import Image_Tools
import matplotlib.pyplot as plt

print(cv2.__version__)


class Dartboard_Detector:
    ENV = {
        'DARTBOARD_SHAPE': (1000, 1000),

        'DETECTION_BLUR': (5, 5),
        'DETECTION_GREEN_LOW': 90,
        'DETECTION_GREEN_HIGH': 120,
        'DETECTION_RED_LOW': 0,
        'DETECTION_RED_HIGH': 40,
        'DETECTION_STRUCTURING_ELEMENT': (100, 100),
        'DETECTION_BINARY_THRESHOLD_MIN': 127,
        'DETECTION_BINARY_THRESHOLD_MAX': 255,
        'DETECTION_OFFSET': 200,

        'ORIENTATION_BLUR': (5, 5),
        'ORIENTATION_COLOR_LOW': 45,
        'ORIENTATION_COLOR_HIGH': 60,
        'ORIENTATION_KERNEL': (100, 100),
        'ORIENTATION_ELEMENT_SIZE_MIN': 350,
        'ORIENTATION_ELEMENT_SIZE_MAX': 600,

        'ORIENTATION_TEMPLATES': ['shape_top.png', 'shape_bottom.png', 'shape_left.png', 'shape_right.png']

    }

    def scaleROI(self, IM):
        if (IM.ndim == 3):
            IM_normal = np.zeros((self.ENV['DARTBOARD_SHAPE'][0], self.ENV['DARTBOARD_SHAPE'][1], IM.shape[2]), "uint8")
        else:
            IM_normal = np.zeros((self.ENV['DARTBOARD_SHAPE'][0], self.ENV['DARTBOARD_SHAPE'][1]), "uint8")
        scale = 1
        if IM.shape[0] > IM.shape[1]:
            # higher than width
            scale = IM_normal.shape[0] / IM.shape[0]
        else:
            # widther than high
            scale = IM_normal.shape[1] / IM.shape[1]
        new_y = int(IM.shape[0] * scale)
        new_x = int(IM.shape[1] * scale)
        offset_y = int((IM_normal.shape[0] - new_y) / 2)
        offset_x = int((IM_normal.shape[1] - new_x) / 2)
        IM_resized = cv2.resize(IM, (new_x, new_y), cv2.INTER_AREA)
        if (IM.ndim == 3):
            IM_normal[offset_y:offset_y + new_y, offset_x:offset_x + new_x, :] = IM_resized
        else:
            IM_normal[offset_y:offset_y + new_y, offset_x:offset_x + new_x] = IM_resized
        return IM_normal

    def detectDartboard(self, IM):
        IM_blur = cv2.blur(IM, Dartboard_Detector.ENV['DETECTION_BLUR'])
        # convert to HSV
        base_frame_hsv = cv2.cvtColor(IM_blur, cv2.COLOR_BGR2HSV)
        # Extract Green
        green_thres_low = int(Dartboard_Detector.ENV['DETECTION_GREEN_LOW'] / 255. * 180)
        green_thres_high = int(Dartboard_Detector.ENV['DETECTION_GREEN_HIGH'] / 255. * 180)
        green_min = np.array([green_thres_low, 100, 100], np.uint8)
        green_max = np.array([green_thres_high, 255, 255], np.uint8)
        frame_threshed_green = cv2.inRange(base_frame_hsv, green_min, green_max)
        # Extract Red
        red_thres_low = int(Dartboard_Detector.ENV['DETECTION_RED_LOW'] / 255. * 180)
        red_thres_high = int(Dartboard_Detector.ENV['DETECTION_RED_HIGH'] / 255. * 180)
        red_min = np.array([red_thres_low, 100, 100], np.uint8)
        red_max = np.array([red_thres_high, 255, 255], np.uint8)
        frame_threshed_red = cv2.inRange(base_frame_hsv, red_min, red_max)
        # Combine
        combined = frame_threshed_red + frame_threshed_green
        # Close
        kernel = np.ones(Dartboard_Detector.ENV['DETECTION_STRUCTURING_ELEMENT'], np.uint8)
        closing = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        # GUI.show(closing, "Dart_Detector")
        # find contours
        ret, thresh = cv2.threshold(combined, Dartboard_Detector.ENV['DETECTION_BINARY_THRESHOLD_MIN'],
                                    Dartboard_Detector.ENV['DETECTION_BINARY_THRESHOLD_MAX'], 0)
        contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        max_cont = -1
        max_idx = 0
        for i in range(len(contours)):
            length = cv2.arcLength(contours[i], True)
            if length > max_cont:
                max_idx = i
                max_cont = length
        x, y, w, h = cv2.boundingRect(contours[max_idx])
        x = x - Dartboard_Detector.ENV['DETECTION_OFFSET']
        y = y - Dartboard_Detector.ENV['DETECTION_OFFSET']
        w = w + int(2 * Dartboard_Detector.ENV['DETECTION_OFFSET'])
        h = h + int(2 * Dartboard_Detector.ENV['DETECTION_OFFSET'])
        #return x, y, w, h, closing, frame_threshed_green, frame_threshed_red
        return 0, 0, IM.shape[1], IM.shape[0], closing, frame_threshed_green, frame_threshed_red

    def getOrientation(self, IM_ROI, IM_ROI_board):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, Dartboard_Detector.ENV['ORIENTATION_KERNEL'])
        # Segment zones
        IM_ROI_blur = cv2.blur(IM_ROI, Dartboard_Detector.ENV['ORIENTATION_BLUR'])
        # convert to HSV
        IM_ROI_HSV = cv2.cvtColor(IM_ROI_blur, cv2.COLOR_BGR2HSV)
        purple_thres_low = int(Dartboard_Detector.ENV['ORIENTATION_COLOR_LOW'] / 255. * 180)
        purple_thres_high = int(Dartboard_Detector.ENV['ORIENTATION_COLOR_HIGH'] / 255. * 180)
        purple_min = np.array([purple_thres_low, 100, 100], np.uint8)
        purple_max = np.array([purple_thres_high, 255, 255], np.uint8)
        frame_thres_color = cv2.inRange(IM_ROI_HSV, purple_min, purple_max)
        # Mask
        frame_thres_color = cv2.subtract(frame_thres_color, IM_ROI_board)
        frame_thres_color_closed = cv2.morphologyEx(frame_thres_color, cv2.MORPH_CLOSE, kernel)

        # Compute contours
        im2, contours, hierarchy = cv2.findContours(frame_thres_color_closed.copy(), cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)
        contour_lengths = []
        contours_structure = []
        for i in range(len(contours)):
            length = cv2.arcLength(contours[i], True)
            contour_lengths.append(length)
            if length > Dartboard_Detector.ENV['ORIENTATION_ELEMENT_SIZE_MIN'] and length < Dartboard_Detector.ENV[
                'ORIENTATION_ELEMENT_SIZE_MAX']:
                contours_structure.append(contours[i])
        # debug histogramm
        # print(len(point_contours))
        # plt.hist(contour_lengths, bins=20, range=(50,1000), normed=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, hold=None, data=None)
        # plt.show()
        return frame_thres_color, frame_thres_color_closed, contours_structure

    def detect_bullseye(self, board):
        blurred = cv2.blur(board,(11,11))
        blurred = cv2.medianBlur(blurred,27)
        ring_cont, hierarchy = cv2.findContours(blurred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        canvas = blurred.copy()
        outer_ring = max(ring_cont, key=Image_Tools.arclen)

        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        outer_ring = cv2.fitEllipse(Image_Tools.scale_contour(outer_ring, 0.68))
        kobi = cv2.cvtColor(blurred.copy(), cv2.COLOR_GRAY2BGR)
        kobi =cv2.drawContours(kobi, ring_cont, -1, (255, 0, 255), 3)
        GUI.imShow_debug(kobi)
        cropped, cropped_bg_black = Image_Tools.crop_ellipse(blurred, outer_ring)
        cropped_bg_black = cv2.cvtColor(cropped_bg_black, cv2.COLOR_BGR2GRAY)
        GUI.imShow_debug(cropped_bg_black)

        better_conts, hierarchy = cv2.findContours(cropped_bg_black, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        noise = Image_Tools.create_noise(cropped_bg_black, 1, 60)
        better_conts += noise
        better_conts, better_conts_cat = Image_Tools.low_cut_unsupervised(better_conts)

        cv2.drawContours(canvas, better_conts, -1, (255, 0, 0), 3)

        min_cont = min(better_conts, key=Image_Tools.arclen)
        cv2.drawContours(canvas, min_cont, -1, (0, 0, 255), 3)
        GUI.imShow_debug(canvas)
        bullseye = cv2.fitEllipse(min_cont)




        center, (d1, d2), angle = bullseye
        cv2.ellipse(canvas, bullseye, (0, 0, 255), thickness=3)
        GUI.imShow_debug(canvas)
        return center

    def oldDetectBullseye(self, board, top, bot, line_size):
        # TODO implement contour instead of selfmade
        xtop, ytop = top
        xbot, ybot = bot
        result = board.copy()
        clone = result.copy()
        clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
        diff = xtop - xbot
        steep = diff / (ytop - ybot)
        i = 0
        color = (255, 255, 255)
        prev = False
        zones = []
        zone_tolerance = 60
        neighbor_tolerance = 12
        #print("Bot" , bot)
        tol = zone_tolerance
        for y in range(int(ytop), int(ybot)):
            for k in range(-line_size, line_size):
                pixel = (int(k + xtop + i * steep), int(y))
                cv2.circle(clone, pixel, 1, (0, 0, 255), thickness=3)
                # print(pixel + " " +´,result[pixel[1], pixel[0]])
                if result[pixel[1], pixel[0]] == 255 and not prev:
                    tol = zone_tolerance
                    prev = True
                    zones.append(pixel)
                    cv2.circle(clone, pixel, 5, (255,0,0), thickness=3)
                elif result[pixel[1], pixel[0]] == 0:
                    tol -= 1
                if tol == 0:
                    prev = False
                    tol = zone_tolerance
            i += 1
        #GUI.imShow(clone)
        #print(len(zones))
        dist = zones[len(zones) - 1][1] - zones[0][1]
        tol_dist = int(dist / neighbor_tolerance)
        #print(tol_dist)
        filtered = []
        before = [-1, -1]

        for pix in zones:
            if pix[1] - before[1] > tol_dist:
                filtered.append(pix)
            before = pix
        bull_start = filtered[2]
        step = abs(int(bull_start[1] - ytop))
        bull_end = 0

        filt_img = board.copy()
        filt_img = cv2.cvtColor(filt_img, cv2.COLOR_GRAY2BGR)
        other = 6
        tol = other
        for pix in filtered:
            cv2.circle(filt_img, pix, 1, (0, 0, 255), thickness=3)

        for y in range(int(bull_start[1]), int(ybot)):

            x = int(xtop + step * steep)
            pixel = int(x), int(y)
            #cv2.circle(result, pixel, 1, (255, 255, 255), thickness=3)
            if result[pixel[1], pixel[0]] == 255:
                tol = other
            elif result[pixel[1], pixel[0]] == 0:
                tol -= 1
            if tol == 0:
                bull_end = pixel
                cv2.circle(filt_img, (int(bull_end[0]), int(bull_end[1])), 3, (0, 255, 255), thickness=3)
                break
            step += 1
        #GUI.imShow(result)
        #print("Here Zones: " + str(len(filtered)))

        add = np.add(bull_start, bull_end)
        arithmetic = np.divide(add, 2)
        cv2.circle(filt_img, (int(arithmetic[0]), int(arithmetic[1])), 40, (255, 255, 255), thickness=3)

        #GUI.imShow(filt_img)

        return arithmetic

    def getOrientationCorr(self, IM_ROI, base_dir):
        kernel_l = cv2.imread(base_dir + self.ENV['ORIENTATION_TEMPLATES'][2])
        kernel_r = cv2.imread(base_dir + self.ENV['ORIENTATION_TEMPLATES'][3])
        kernel_t = cv2.imread(base_dir + self.ENV['ORIENTATION_TEMPLATES'][0])
        kernel_b = cv2.imread(base_dir + self.ENV['ORIENTATION_TEMPLATES'][1])
        temp = cv2.imread(base_dir + "temp.png")
        h = kernel_l.shape[0]
        w = kernel_l.shape[1]

        # right
        res = cv2.matchTemplate(IM_ROI, kernel_r, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        right_top_left = max_loc
        right = (right_top_left[0] + w, right_top_left[1] + h // 2)

        # GUI.imShow(kernel_r)

        # left
        res = cv2.matchTemplate(IM_ROI, kernel_l, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        left_top_left = max_loc
        left = (left_top_left[0], left_top_left[1] + h // 2)
        # GUI.imShow(kernel_l)

        h = kernel_t.shape[0]
        w = kernel_t.shape[1]
        # top
        res = cv2.matchTemplate(IM_ROI, kernel_t, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_top_left = max_loc
        top = (top_top_left[0] + w // 2, top_top_left[1])
        # GUI.imShow(kernel_t)
        GUI.imShow_debug(res)
        # print(max_loc)

        # bottom
        res = cv2.matchTemplate(IM_ROI, kernel_b, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        bottom_top_left = max_loc
        bottom = (bottom_top_left[0] + w // 2, bottom_top_left[1] + h)
        # GUI.imShow(kernel_b)

        IM_ROI_copy = IM_ROI.copy()
        w = kernel_t.shape[1]
        h = kernel_t.shape[0]
        bottom_right = (top_top_left[0] + w, top_top_left[1] + h)
        cv2.rectangle(IM_ROI_copy, top_top_left, bottom_right, 255, 2)
        print(top, bottom, left, right)
        # GUI.imShow(IM_ROI_copy)

        return top_top_left, bottom_top_left, left_top_left, right_top_left, top, bottom, left, right

