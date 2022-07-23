import math

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

from dartboard_detector import Dartboard_Detector
from gui import GUI
from image_tools import Image_Tools


class Dartboard:

    def __init__(self):
        self.warp_mode = ""

    def get_orientation(self, no_arrow_roi):

        rings = self.get_rings(no_arrow_roi, blur=9)
        # GUI.imShow(rings)

        ellipse = self.fit_outer_ellipse(rings)
        center, (d1, d2), angle = ellipse
        xc, yc = center
        # draw ellipse
        result = cv.cvtColor(rings.copy(), cv.COLOR_GRAY2BGR)
        cv.ellipse(result, ellipse, (0, 255, 0), thickness=3)
        cv.circle(result, (int(xc), int(yc)), 10, (255, 0, 0), -1)
        GUI.imShow(result)

        # mask = np.zeros((rings.shape[0], rings.shape[1], 3), dtype='uint8')
        # bg = np.zeros((rings.shape[0], rings.shape[1], 3), dtype='uint8')
        # bg[:] = (0, 0, 255)
        # mask[:] = (0, 0, 0)
        # rings = cv.cvtColor(rings, cv.COLOR_GRAY2BGR)
        # cv.ellipse(mask, ellipse, (255, 255, 255), thickness=-1)
        # cropped = cv.bitwise_and(mask, rings.copy())
        # circle = cv.bitwise_not(mask, bg)
        # # GUI.imShow(circle)
        # cropped = cv.bitwise_or(circle, cropped)
        #
        #
        # mask[:] = (255, 255, 255)
        # cv.ellipse(mask, ellipse, (0, 0, 0), thickness=-1)
        # circle = cv.bitwise_not(mask, bg)
        # cropped_bg_black = cv.bitwise_and(circle, cropped)
        # GUI.imShow(cropped_bg_black)

        rmajor = max(d1, d2) / 2
        rminor = min(d1, d2) / 2
        if angle > 90:
            angle = angle - 90
        else:
            angle = angle + 90

        top, bot = self.calc_border_points(center, (rminor, rmajor), angle)

        dartboard_detector = Dartboard_Detector()
        # bullseye = dartboard_detector.oldDetectBullseye(rings, top, bot, 10)
        # cropped_bg_black = cv.cvtColor(cropped_bg_black, cv.COLOR_BGR2GRAY)
        # rings = cv.cvtColor(rings, cv.COLOR_BGR2GRAY)
        bullseye = dartboard_detector.detect_bullseye(rings)

        dist_thresh = 1.25
        if yc / bullseye[1] > dist_thresh:
            self.warp_mode = "down"
        else:
            self.warp_mode = "up"

        right_pt = self.calc_dst_point(center, (rminor, rmajor), bullseye, (top, bot))
        left_pt = [xc - (right_pt[0] - xc), right_pt[1]]

        new_angle = 90 - math.atan2(bot[1] - top[1], bot[0] - top[0]) * 180 / math.pi

        rot_right_pt = self.rotate_point(right_pt, center, new_angle)
        rot_left_pt = self.rotate_point(left_pt, center, new_angle)

        top = (int(top[0]), int(top[1]))
        bot = (int(bot[0]), int(bot[1]))
        left = (int(rot_left_pt[0]), int(rot_left_pt[1]))
        right = (int(rot_right_pt[0]), int(rot_right_pt[1]))

        blue = (255, 0, 0)
        white = (255, 255, 255)
        green = (0, 255, 0)
        yellow = (0, 255, 255)

        # top_neu = bot
        # bot_neu = top
        # left_neu = right
        # right_neu = left

        bruh = no_arrow_roi.copy()
        cv.circle(bruh, top, 10, green, -1)
        cv.circle(bruh, (int(left_pt[0]), int(left_pt[1])), 10, green, -1)
        cv.circle(bruh, (int(right_pt[0]), int(right_pt[1])), 10, green, -1)
        cv.circle(bruh, (left), 10, white, -1)
        cv.circle(bruh, (right), 10, blue, -1)
        cv.circle(bruh, bot, 10, yellow, -1)
        cv.circle(bruh, (int(xc), int(yc)), 10, yellow, -1)
        cv.circle(bruh, (int(bullseye[0]), int(bullseye[1])), 10, (255, 255, 0), -1)
        cv.line(bruh, top, bot, blue, thickness=3)
        GUI.imShow(bruh)

        border_margin = 50
        target_dim = 1000

        # source = np.float32(
        #     [[bullseye[0], bullseye[1]],[bot_neu[0], bot_neu[1]], [right_neu[0], right_neu[1]], [left_neu[0], left_neu[1]],
        #      [top_neu[0], top_neu[1]]])
        source = np.float32(
            [[bullseye[0], bullseye[1]], [right[0], right[1]], [left[0], left[1]],
             [bot[0], bot[1]],
             [top[0], top[1]]])

        half = int(target_dim / 2)
        dest = np.float32(
            [[half, half], [target_dim - border_margin, half], [0 + border_margin, half],
             [half, target_dim - border_margin],
             [half, 0 + border_margin]])

        mot, status = cv.findHomography(source, dest)
        return mot

    def get_rings(self, no_arrow_roi, blur):
        dartboard_detector = Dartboard_Detector()
        x, y, w, h, BOARD, GREEN, RED = dartboard_detector.detectDartboard(no_arrow_roi)
        rings = cv.bitwise_or(RED, GREEN)
        rings = cv.medianBlur(rings, blur)
        return rings

    def fit_outer_ellipse(self, rings):
        # blur = int(rings.shape[1] / 100)
        # print(blur)
        # if blur % 2 == 0:
        #     blur += 3

        rings = cv.blur(rings, (11, 11))
        GUI.imShow_debug(rings)
        contours = cv.findContours(rings, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        bo = cv.cvtColor(rings.copy(), cv.COLOR_GRAY2BGR)

        # TODO fix wrong contour choosing
        big_contour = max(contours, key=Image_Tools.arclen)
        cv.drawContours(bo, big_contour, -1, (255, 0, 0), 3)
        # fit contour to ellipse and get ellipse center, minor and major diameters and angle in degree
        GUI.imShow_debug(bo)
        ellipse = cv.fitEllipse(big_contour)
        return ellipse

    def calc_border_points(self, center, radii, angle):

        xc, yc = center
        rminor, rmajor = radii

        xtop = xc + math.cos(math.radians(angle)) * rmajor
        ytop = yc + math.sin(math.radians(angle)) * rmajor
        xbot = xc + math.cos(math.radians(angle + 180)) * rmajor
        ybot = yc + math.sin(math.radians(angle + 180)) * rmajor

        yright = yc + math.sin(math.radians(angle + 90)) * rminor
        xright = xc + math.cos(math.radians(angle + 90)) * rminor
        yleft = yc + math.sin(math.radians(angle + 270)) * rminor
        xleft = xc + math.cos(math.radians(angle + 270)) * rminor

        diff_min = (yleft - yright) ** 2
        diff_max = (ytop - ybot) ** 2

        if diff_min > diff_max:

            if yleft > yright:
                top = (xright, yright)
                bot = (xleft, yleft)

            else:
                top = (xleft, yleft)
                bot = (xright, yright)
        else:
            # print("ASda")
            top = (xtop, ytop)
            bot = (xbot, ybot)

        return top, bot

    def calc_dst_point(self, center, radii, bullseye, src_pts):

        xc, yc = center
        b, a = radii
        top, bot = src_pts

        # xtop, ytop = top
        # xbot, ybot = bot
        # # diff = xtop - xbot
        # # steep = diff / (ytop - ybot)

        travel = bullseye[1] - yc
        # print("T", travel)
        new_y = yc + travel
        # print("ney ", travel)
        res = math.sqrt((1 - travel ** 2 / b ** 2) * a ** 2) + xc
        return [res, new_y]

    def rotate_point(self, pt, center, angle):

        xc, yc = center
        rot_mat = cv.getRotationMatrix2D((xc, yc), angle, 1.0)
        mat_2d = rot_mat[:2, :2]

        z = np.array([xc, yc])
        p = np.array([pt[0], pt[1]])
        zp = np.subtract(p, z)

        q = np.matmul(mat_2d, zp)
        end_cor = np.add(z, q)

        return end_cor

    def warp_board(self, im_roi, mot, target_dim):
        warped_board = cv.warpPerspective(im_roi.copy(), mot, (target_dim, target_dim), flags=cv.INTER_AREA)
        return warped_board

    def get_white_contours(self, cropped):
        GUI.imShow_debug(cropped)
        brightness = Image_Tools.get_brightness(cropped)
        blurred_cropped = cv.blur(cropped, (5, 5))
        blurred_cropped_hsv = cv.cvtColor(blurred_cropped, cv.COLOR_BGR2HSV)


        print("Brit: ", brightness)
        base_brit = 85


        change_val = (brightness - base_brit) * 1.2

        value_min = 183 + change_val
        white_zone_len = 0
        step_size = 1
        steps = 0
        step_max = 100
        after_proc_done = False
        over = -1
        under = -1
        prev = 0
        minus = True
        check_point = 0
        while steps < step_max and not after_proc_done:
            if white_zone_len != 20:
                if prev > white_zone_len and check_point == 0:
                    minus = not minus
                    check_point = 5
                if minus:
                    value_min -= step_size + (step_max-steps * 0.5)
                else:
                    value_min +=step_size + (step_max-steps * 0.5)
                print("Len ", white_zone_len, " value ", value_min)
                prev = white_zone_len
            else:
                after_proc_done = True

            black_min = np.array([0, 0, value_min], np.uint8)
            black_max = np.array([180, 100, 255], np.uint8)
            frame_threshed_black = cv.inRange(blurred_cropped_hsv, black_min, black_max)

            blp = cv.medianBlur(frame_threshed_black, 17)

            white_zone_contours, hierarchy = cv.findContours(blp.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            white_zone_contours += Image_Tools.create_noise(cropped, 4, 5)
            white_zones, white_in_cat = Image_Tools.low_cut(white_zone_contours, 3)
            white_zone_len = len(white_zones)

            steps += 1
            if check_point != 0:
                check_point-=1
        kok = cv.drawContours(cv.cvtColor(blp.copy(), cv.COLOR_GRAY2BGR), white_zones, -1, (0, 0, 255), 2)


        GUI.imShow_debug(kok)
        return white_zones, white_in_cat

    def getas_white_contours(self, cropped):
        GUI.imShow_debug(cropped)
        brightness = Image_Tools.get_brightness(cropped)
        blurred_cropped = cv.blur(cropped, (5, 5))
        blurred_cropped_hsv = cv.cvtColor(blurred_cropped, cv.COLOR_BGR2HSV)


        #print("Brit: ", brightness)

        value_min = 183
        black_zone_len = 0
        step_size = 1
        steps = 0
        step_max = 100
        after_proc_done = False
        over = -1
        under = -1

        black_min = np.array([0, 0, value_min], np.uint8)
        black_max = np.array([180, 100, 255], np.uint8)
        frame_threshed_black = cv.inRange(blurred_cropped_hsv, black_min, black_max)

        blp = cv.medianBlur(frame_threshed_black, 17)

        white_zone_contours, hierarchy = cv.findContours(blp.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        white_zone_contours += Image_Tools.create_noise(cropped, 4, 5)
        white_zones, white_in_cat = Image_Tools.low_cut(white_zone_contours, 3)


        kok = cv.drawContours(cv.cvtColor(blp.copy(), cv.COLOR_GRAY2BGR), white_zones, -1, (0, 0, 255), 2)


        GUI.imShow_debug(kok)
        return white_zones, white_in_cat

    def get_red_mask(self, warped_board):
        DETECTION_RED_LOW = 0
        DETECTION_RED_HIGH = 40

        blurred_board = cv.blur(warped_board, (5, 5))
        blurred_board_hsv = cv.cvtColor(blurred_board, cv.COLOR_BGR2HSV)

        red_thres_low = int(DETECTION_RED_LOW / 255. * 180)
        red_thres_high = int(DETECTION_RED_HIGH / 255. * 180)
        red_min = np.array([red_thres_low, 100, 100], np.uint8)
        red_max = np.array([red_thres_high, 255, 255], np.uint8)
        frame_threshed_red = cv.inRange(blurred_board_hsv, red_min, red_max)

        blurred_red = cv.medianBlur(frame_threshed_red, 9)

        return blurred_red

    def get_green_mask(self, warped_board):
        DETECTION_GREEN_LOW = 90
        DETECTION_GREEN_HIGH = 120

        blurred_board = cv.blur(warped_board, (5, 5))
        blurred_board_hsv = cv.cvtColor(blurred_board, cv.COLOR_BGR2HSV)

        green_thres_low = int(DETECTION_GREEN_LOW / 255. * 180)
        green_thres_high = int(DETECTION_GREEN_HIGH / 255. * 180)
        green_min = np.array([green_thres_low, 100, 100], np.uint8)
        green_max = np.array([green_thres_high, 255, 255], np.uint8)
        frame_threshed_green = cv.inRange(blurred_board_hsv, green_min, green_max)

        blurred_green = cv.medianBlur(frame_threshed_green, 13)
        return blurred_green

    def get_black_contours(self, cropped):
        GUI.imShow_debug(cropped)
        brightness = Image_Tools.get_brightness(cropped)
        blurred_cropped = cv.blur(cropped, (5, 5))
        blurred_cropped_hsv = cv.cvtColor(blurred_cropped, cv.COLOR_BGR2HSV)


        #print("Brit: ", brightness)
        base_brit = 181
        saturation_max = 255

        change_val = (brightness - base_brit) * 1.2

        value_max = 141 + change_val
        black_zone_len = 0
        step_size = 1
        steps = 0
        step_max = 100
        after_proc_done = False
        over = -1
        under = -1

        while steps < step_max and not after_proc_done:
            if black_zone_len != 20:
                value_max -= step_size
            else:
                after_proc_done = True


            black_min = np.array([0, 0, 0], np.uint8)
            black_max = np.array([180, saturation_max, value_max], np.uint8)
            frame_threshed_black = cv.inRange(blurred_cropped_hsv, black_min, black_max)

            blp = cv.medianBlur(frame_threshed_black, 17)

            black_zone_contours, hierarchy = cv.findContours(blp.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

            black_zone_contours += Image_Tools.create_noise(cropped, 4, 5)
            black_zones, black_in_cat = Image_Tools.low_cut(black_zone_contours, 3)
            black_zone_len = len(black_zones)

            steps += 1
        kok = cv.drawContours(cv.cvtColor(blp.copy(), cv.COLOR_GRAY2BGR), black_zones, -1, (0, 0, 255), 2)
        over = 0
        under = 0

        GUI.imShow_debug(kok)
        return black_zones, black_in_cat

    def crop_board(self, warped_board, red_mask, green_mask, scale):
        rings = cv.blur(cv.bitwise_or(green_mask, red_mask), (59, 59))
        ring_cont, hierarchy = cv.findContours(rings, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # kol = cv.drawContours(rings, ring_cont, -1, (0, 255, 255), 2)
        # GUI.imShow(kol)
        big_contour = max(ring_cont, key=cv.contourArea)
        big_contour = Image_Tools.scale_contour(big_contour, scale)

        ellipse = cv.fitEllipse(big_contour)

        cropped, cropped_bg_black = Image_Tools.crop_ellipse(warped_board, ellipse)

        return cropped, cropped_bg_black

    def crop_new(self, warped):

        blurred_cropped_hsv = cv.cvtColor(warped.copy(), cv.COLOR_BGR2HSV)

        saturation_max = 255
        value_max = 170

        black_min = np.array([0, 0, 0], np.uint8)
        black_max = np.array([179, saturation_max, value_max], np.uint8)
        frame_threshed_black = cv.inRange(blurred_cropped_hsv, black_min, black_max)
        frame_threshed_black = cv.medianBlur(frame_threshed_black, 11)
        circles = cv.HoughCircles(frame_threshed_black, cv.HOUGH_GRADIENT, 1.5, 100)

        # cv.drawContours(kol, sorted_conts[0], -1, (0, 0, 255),2)
        big_circle = (5, 5, 5)
        output = warped.copy()
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            # loop over the (x, y) coordinates and radius of the circles
            big_circle = max(circles, key=Image_Tools.circle_area)
            x, y, r = big_circle
            cv.circle(output, (x, y), r, (255, 0, 0), 3)
            # show the output image
            # GUI.im_compare([frame_threshed_black, output])
        x, y, r = big_circle
        canvas = np.zeros((output.shape[0], output.shape[1]), dtype='uint8')
        cv.circle(canvas, (x, y), r, (255, 255, 255), thickness=-1)
        GUI.imShow_debug(canvas)

        ring_cont, hierarchy = cv.findContours(canvas, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        outer = max(ring_cont, key=cv.contourArea)
        ellipse = cv.fitEllipse(outer)
        cropped, cropped_black = Image_Tools.crop_ellipse(warped, ellipse)

        return cropped, cropped_black

    def draw_contours(self, warped_board, white, black, red, green):
        canvas = warped_board.copy()
        red_c = (0, 0, 255)
        blue_c = (255, 0, 0)
        yellow_c = (0, 255, 255)
        green_c = (0, 255, 0)

        cv.drawContours(canvas, white[1], -1, red_c, 2)
        cv.drawContours(canvas, black[1], -1, green_c, 2)
        cv.drawContours(canvas, red[1], -1, blue_c, 2)
        cv.drawContours(canvas, green[1], -1, yellow_c, 2)

        return canvas

    def calc_over_and_under(self, cont_in_cat, threshold):
        under = 0
        over = 0
        for conts in cont_in_cat:
            areas = [cv.contourArea(cont) for cont in conts]
            areas_co = [[cv.contourArea(cont), i*0.01] for i,cont in enumerate(conts)]
            cluster = AgglomerativeClustering(distance_threshold=300.0, affinity='euclidean', linkage='single', n_clusters=None)
            cluster.fit_predict(areas_co)
            plt.scatter(areas, list(range(len(areas))), c=cluster.labels_, cmap='rainbow')
            plt.show()
            mean = np.mean(areas)
            for cont in conts:
                if cv.contourArea(cont) - mean < -threshold:
                    under += 1
                elif cv.contourArea(cont) - mean > threshold:
                    over += 1
        return over, under
