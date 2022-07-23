import math
import sys

from alive_progress import alive_bar
from matplotlib import pylab
from sklearn.cluster import KMeans

from gui import GUI
import numpy as np
import cv2 as cv
from image_tools import Image_Tools
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


class Calcboard:
    def __init__(self, white_in_cat, black_in_cat, red_mask, green_mask, warped, warp_mode):
        self.white_in_cat = white_in_cat
        self.black_in_cat = black_in_cat
        self.red_mask = red_mask
        self.green_mask = green_mask
        self.warp_mode = warp_mode
        self.warped = warped
        self.NUMBERS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

        self.outer_big_merged = []
        self.inner_big_merged = []

        self.outer_ring_merged = []
        self.inner_ring_merged = []

        self.generate_board(warped)

    def sort_contours(self, contours):

        if len(contours) == 0:
            print("Fehlerhafte Feldererkennung")
            return

        use_list = contours
        sorted_list = []

        blank = np.zeros((1000, 1000, 3), dtype='uint8')
        starting_point = 0
        prev = 10 ** 10
        for i, contour in enumerate(use_list):
            cx, cy = Image_Tools.get_contour_center(contour)
            distance = abs(500 - cx)
            if distance < prev and cy < 500:
                prev = distance
                starting_point = i

        closest = use_list[starting_point]
        closest_num = starting_point
        cv.drawContours(blank, closest, -1, (255, 255, 255), 5)
        # GUI.imShow(blank)
        com_nums = []

        for contour in use_list:
            prev_dist = 10 ** 10
            cx, cy = Image_Tools.get_contour_center(closest)
            sorted_list.append(closest)
            com_nums.append(closest_num)
            for i, comparison in enumerate(use_list):
                rx, ry = Image_Tools.get_contour_center(comparison)
                if (rx == cx and ry == cy) or (len(com_nums) == 1 and rx < cx):
                    continue
                distance = math.sqrt((cx - rx) ** 2 + (ry - cy) ** 2)
                if distance < prev_dist and i not in com_nums:
                    prev_dist = distance
                    closest_num = i
                    closest = comparison
        blank = np.zeros((1000, 1000, 3), dtype='uint8')

        red_c = (0, 0, 255)
        blue_c = (255, 0, 0)
        yellow_c = (0, 255, 255)
        counter = 1
        for cont in sorted_list:
            if counter == 1:
                color = red_c
                counter += 1
            elif counter == 2:
                color = blue_c
                counter += 1
            elif counter == 3:
                color = yellow_c
                counter = 1
            cv.drawContours(blank, cont, -1, color, 2)

        # GUI.imShow(blank)
        return sorted_list

    def calc_ring_contours(self, inner_contours, outer_contours, warped_board):
        LINE_UP = '\033[1A'
        LINE_CLEAR = '\x1b[2K'
        inner_ring = self.get_inner_ring()
        outer_ring = self.get_outer_ring()
        inner_ring_conts = []
        outer_ring_conts = []

        canvas = cv.cvtColor(warped_board.copy(), cv.COLOR_BGR2GRAY)
        canvas = cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)

        # n_row = int(2)
        # _, axs = plt.subplots(2, 2)
        # axs = axs.flatten()
        cords = [[0, 500], [0, 500]]
        prev_cords = cords
        # print("0% Felderkennnung")
        with alive_bar(len(inner_contours), title="Felderkennung") as bar:
            for k, contour in enumerate(inner_contours):
                bar()

                asa = np.zeros((1000, 1000), dtype='uint8')
                canvas = cv.cvtColor(warped_board.copy(), cv.COLOR_BGR2GRAY)
                canvas = cv.cvtColor(canvas, cv.COLOR_GRAY2BGR)

                color = (255, 255, 255)

                # print(prefix, ": ", real)
                cv.drawContours(asa, [contour], 0, color, 1)
                # asa = cv.cvtColor(asa, cv.COLOR_BGR2GRAY)
                # black_con = black_in_cat[k]

                color = (255, 255, 255)
                # cv.drawContours(asa, [black_con], 0, color, 2)

                dst = cv.Canny(asa, 50, 200, None, 3)

                lines = cv.HoughLines(dst, 1, np.pi / 180, 60, None, 0, 0)

                prev = ""
                # color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
                color = (0, 0, 255)
                cv.putText(canvas, str(k), Image_Tools.get_contour_center(contour), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                           cv.LINE_AA)
                acba = np.zeros(warped_board.shape, dtype='uint8')

                raw_lines = []
                steeps = []

                if lines is not None:
                    # print("-------------- Neue Lines -------------")
                    for i in range(0, len(lines)):
                        rho = lines[i][0][0]
                        theta = lines[i][0][1]
                        a = math.cos(theta)
                        b = math.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                        # cv.line(canvas, pt1, pt2, color, 1, cv.LINE_AA)
                        if pt1[1] - pt2[1] == 0 or pt1[0] - pt2[0] == 0:
                            continue

                        steep = (pt1[1] - pt2[1]) / (pt1[0] - pt2[0])
                        y_cut = pt1[1] - (pt1[0] * steep)

                        steeps.append(steep)
                        # cv.line(canvas, pt1, pt2, color, 1, cv.LINE_AA)
                        raw_lines.append([y_cut, steep])

                steeps_i = [[steep, i * 0.001] for i, steep in enumerate(steeps)]

                kmeans = KMeans(n_clusters=2, random_state=4).fit(steeps_i)
                categories = kmeans.labels_

                sorted_lines = [[], []]
                for i, line in enumerate(raw_lines):
                    sorted_lines[categories[i]].append(line)
                # print(raw_lines)
                # print(categories)
                # print(sorted_lines)
                line1 = sorted_lines[0][0]
                line2 = sorted_lines[1][0]

                mean_lines = []
                for line_group in sorted_lines:
                    mod_steep = []
                    mod_cut = []
                    for line in line_group:
                        mod_steep.append(line[1])
                        mod_cut.append(line[0])

                    steep_mean = np.median(mod_steep)
                    cut_mean = np.median(mod_cut)
                    # print("Steep ", steep_mean, " cut ", cut_mean)
                    mean_lines.append([cut_mean, steep_mean])

                oclock = False
                points = []

                if k == 5 or k == 15:
                    cords = prev_cords
                else:
                    cords = self.calc_cords(mean_lines, k)
                    prev_cords = cords
                for l, line in enumerate(mean_lines):
                    x1 = cords[l][0]
                    x2 = cords[l][1]

                    cut = line[0]
                    steep = line[1]

                    pt1 = [int(x1), int(x1 * steep + cut)]
                    pt2 = [int(x2), int(x2 * steep + cut)]

                    cv.line(canvas, pt1, pt2, (255, 0, 0), 5, cv.LINE_AA)
                    if l != 0:
                        intersect_x = (cut - prev_cut) / (prev_steep - steep)
                        intersect_point = [int(intersect_x), int(intersect_x * steep + cut)]
                        points.append(intersect_point)
                    if x1 == 0:
                        points.append(pt1)
                    else:
                        points.append(pt2)

                    prev_steep = steep
                    prev_cut = cut

                cropped = acba
                # print(points)
                points = np.array(points)
                tri_cut = np.zeros((1000, 1000), dtype='uint8')
                cv.fillPoly(tri_cut, pts=[points], color=(255, 255, 255))
                # canvas = cv.bitwise_xor(canvas, cropped)
                # fig = pylab.gcf()
                # fig.canvas.manager.set_window_title(str(k))
                # GUI.imShow(tri_cut)

                new_cont_frame = cv.bitwise_and(inner_ring, tri_cut)
                new_cont_frame_big = cv.bitwise_and(outer_ring, tri_cut)
                new_cont, hierachy = cv.findContours(new_cont_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                new_cont_big, hierachy = cv.findContours(new_cont_frame_big, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

                inner_ring_conts += new_cont
                outer_ring_conts += new_cont_big

                cv.drawContours(canvas, new_cont, 0, (255, 0, 0), 3)
                cv.drawContours(canvas, new_cont_big, 0, (255, 0, 0), 3)

                # axs[0].imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
                # axs[2].imshow(cv.cvtColor(canvas, cv.COLOR_BGR2RGB))
                # axs[3].imshow(cv.cvtColor(tri_cut, cv.COLOR_BGR2RGB))
                # axs[1].scatter(steeps, list(range(len(steeps))), cmap='rainbow', c=categories)
                #
                # plt.pause(1.5)
                # axs[1].clear()

        # plt.show()

        return inner_ring_conts, outer_ring_conts

    def generate_board(self, warped_board):
        big_outer = self.white_in_cat[1] + self.black_in_cat[1]
        big_inner = self.white_in_cat[0] + self.black_in_cat[0]
        self.get_outer_ring_new()
        # red, red_in_cat = Image_Tools.low_cut(self.red, 3)
        # green, green_in_cat = Image_Tools.low_cut(self.green, 3)
        # outer_rings = green_in_cat[1] + red_in_cat[1]
        # inner_rings = red_in_cat[0] + green_in_cat[0]
        blank = np.zeros((1000, 1000, 3), dtype='uint8')

        red_c = (0, 0, 255)
        blue_c = (255, 0, 0)
        yellow_c = (0, 255, 255)
        green = (0, 255, 0)

        outer_sort = self.sort_contours(big_outer)
        inner_sort = self.sort_contours(big_inner)

        new_man = np.zeros((warped_board.shape[1], warped_board.shape[0]), dtype='uint8')

        cv.drawContours(new_man, big_outer, -1, (255, 255, 255), -1)
        cv.drawContours(new_man, big_inner, -1, (255, 255, 255), -1)

        greok = cv.medianBlur(new_man, 21)
        greok = cv.blur(greok, (7, 7))

        akoba, hierarchy = cv.findContours(greok, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        cop = np.zeros((new_man.shape[1], new_man.shape[0], 1), dtype='uint8')

        for i in range(5):
            cx = np.random.randint(cop.shape[1])
            cy = np.random.randint(cop.shape[0])
            cv.circle(cop, (cx, cy), 4, (255), -1)

        noise, hierachy = cv.findContours(cop, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        akoba += noise

        akoba_fil, akoba_in_cat = Image_Tools.low_cut(akoba, 5)

        sort_akoba = sorted(akoba_fil, key=cv.contourArea)
        sort_akoba.remove(sort_akoba[len(sort_akoba) - 1])
        sort_akoba.remove(sort_akoba[0])

        greay = cv.cvtColor(warped_board, cv.COLOR_BGR2GRAY)

        paint = np.zeros((warped_board.shape[1], warped_board.shape[0]), dtype='uint8')
        first = paint.copy()
        second = paint.copy()

        ellips = cv.fitEllipse(sort_akoba[0])
        ellipa = cv.fitEllipse(sort_akoba[1])

        # cyman = cv.drawContours(paint, sort_akoba, -1, (255, 255, 255), 2)

        # cv.ellipse(first, ellips, (255, 255, 255), thickness=-1)
        # cv.ellipse(second, ellipa, (255, 255, 255), thickness=-1)

        # cyman = cv.bitwise_xor(first, second)
        small_filter = Image_Tools.scale_contour(sort_akoba[1], scale=1.1)

        filtered_outer_rings = []
        filtered_inner_rings = []

        # for contour in outer_rings:
        #     center = Image_Tools.get_contour_center(contour)
        #     is_in_contour = cv.pointPolygonTest(small_filter, center, False)
        #     if is_in_contour == 1 or is_in_contour == 0:
        #         filtered_inner_rings.append(contour)
        #     else:
        #         filtered_outer_rings.append(contour)
        # for contour in inner_rings:
        #     center = Image_Tools.get_contour_center(contour)
        #     is_in_contour = cv.pointPolygonTest(small_filter, center, False)
        #     # cv.drawContours(blank, contour, -1, (255,255,255), 2)
        #     # cv.circle(blank, center, 5, (0, 0, 255), -1)
        #     if is_in_contour == -1:
        #         filtered_outer_rings.append(contour)
        #     else:
        #         filtered_inner_rings.append(contour)
        #     # print(filtered_outer_rings)

        self.outer_big_merged = outer_sort
        self.inner_big_merged = inner_sort

        filtered_inner_rings, filtered_outer_rings = self.calc_ring_contours(inner_sort, outer_sort, warped_board)

        cv.drawContours(blank, outer_sort, -1, red_c, 2)
        cv.drawContours(blank, inner_sort, -1, blue_c, 2)
        cv.drawContours(blank, filtered_outer_rings, -1, yellow_c, 2)
        cv.drawContours(blank, filtered_inner_rings, -1, green, 2)
        # cv.drawContours(blank, small_filter, -1, blue_c, 2)
        self.outer_ring_merged = self.sort_contours(filtered_outer_rings)
        self.inner_ring_merged = self.sort_contours(filtered_inner_rings)

        GUI.imShow(blank)

    def merge_with_nums(self, contours):
        merged_list = []
        for i, contour in enumerate(contours):
            merged_list.append([contours, self.NUMBERS[i]])
        return merged_list

    def calc_score(self, arrow_x, arrow_y):
        if self.outer_big_merged is not None:
            for i, contour in enumerate(self.outer_big_merged):
                is_in_contour = cv.pointPolygonTest(contour, (arrow_x, arrow_y), False)
                if is_in_contour == 1 or is_in_contour == 0:
                    return self.NUMBERS[i]

        if self.inner_big_merged is not None:
            for i, contour in enumerate(self.inner_big_merged):
                is_in_contour = cv.pointPolygonTest(contour, (arrow_x, arrow_y), False)
                if is_in_contour == 1 or is_in_contour == 0:
                    return self.NUMBERS[i]

        if self.outer_ring_merged is not None:
            for i, contour in enumerate(self.outer_ring_merged):
                is_in_contour = cv.pointPolygonTest(contour, (arrow_x, arrow_y), False)
                if is_in_contour == 1 or is_in_contour == 0:
                    return self.NUMBERS[i] * 2

        if self.inner_ring_merged is not None:
            for i, contour in enumerate(self.inner_ring_merged):
                is_in_contour = cv.pointPolygonTest(contour, (arrow_x, arrow_y), False)
                if is_in_contour == 1 or is_in_contour == 0:
                    return self.NUMBERS[i] * 3

        return "Nicht im Feld"

    def get_inner_ring(self):
        big_outer = self.white_in_cat[1] + self.black_in_cat[1]
        big_inner = self.white_in_cat[0] + self.black_in_cat[0]

        new_man = np.zeros((1000, 1000), dtype='uint8')

        cv.drawContours(new_man, big_outer, -1, (255, 255, 255), -1)
        cv.drawContours(new_man, big_inner, -1, (255, 255, 255), -1)

        greok = cv.medianBlur(new_man, 21)
        greok = cv.blur(greok, (7, 7))

        akoba, hierarchy = cv.findContours(greok, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        cop = np.zeros((new_man.shape[1], new_man.shape[0], 1), dtype='uint8')

        for i in range(5):
            cx = np.random.randint(cop.shape[1])
            cy = np.random.randint(cop.shape[0])
            cv.circle(cop, (cx, cy), 4, (255), -1)

        noise, hierachy = cv.findContours(cop, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        akoba += noise

        akoba_fil, akoba_in_cat = Image_Tools.low_cut(akoba, 5)

        sort_akoba = sorted(akoba_fil, key=cv.contourArea)
        sort_akoba.remove(sort_akoba[len(sort_akoba) - 1])
        sort_akoba.remove(sort_akoba[0])

        paint = np.zeros((1000, 1000), dtype='uint8')
        first = paint.copy()
        second = paint.copy()

        ellips = cv.fitEllipse(sort_akoba[0])
        ellipa = cv.fitEllipse(sort_akoba[1])

        # cyman = cv.drawContours(paint, sort_akoba, -1, (255, 255, 255), 2)

        cv.ellipse(first, ellips, (255, 255, 255), thickness=-1)
        cv.ellipse(second, ellipa, (255, 255, 255), thickness=-1)

        cyman = cv.bitwise_xor(first, second)

        return cyman

    def get_outer_ring(self):
        big_outer = self.white_in_cat[1] + self.black_in_cat[1]
        big_inner = self.white_in_cat[0] + self.black_in_cat[0]

        new_man = np.zeros((1000, 1000), dtype='uint8')

        cv.drawContours(new_man, big_outer, -1, (255, 255, 255), -1)
        cv.drawContours(new_man, big_inner, -1, (255, 255, 255), -1)

        greok = cv.medianBlur(new_man, 21)
        greok = cv.blur(greok, (7, 7))

        akoba, hierarchy = cv.findContours(greok, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        cop = np.zeros((new_man.shape[1], new_man.shape[0], 1), dtype='uint8')

        doa = cv.cvtColor(new_man.copy(), cv.COLOR_GRAY2BGR)

        cv.drawContours(doa, akoba, -1, (255, 0, 0), 5)

        for i in range(5):
            cx = np.random.randint(cop.shape[1])
            cy = np.random.randint(cop.shape[0])
            cv.circle(cop, (cx, cy), 4, (255), -1)

        noise, hierachy = cv.findContours(cop, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        akoba += noise

        akoba_fil, akoba_in_cat = Image_Tools.low_cut(akoba, 5)

        sort_akoba = sorted(akoba_fil, key=cv.contourArea)
        sort_akoba.remove(sort_akoba[0])

        paint = np.zeros((1000, 1000), dtype='uint8')
        first = paint.copy()
        second = paint.copy()

        ellips = cv.fitEllipse(sort_akoba[1])
        ellipa = cv.fitEllipse(sort_akoba[2])

        # cyman = cv.drawContours(paint, sort_akoba, -1, (255, 255, 255), 2)

        cv.ellipse(first, ellips, (255, 255, 255), thickness=-1)
        cv.ellipse(second, ellipa, (255, 255, 255), thickness=-1)

        rings = cv.bilateralFilter(cv.bitwise_or(self.green_mask, self.red_mask), 25, 10000, 75)
        rings = cv.medianBlur(rings, 11)

        #GUI.imShow(rings)
        # rings = cv.blur(cv.bitwise_or(self.green_mask, self.red_mask), (59, 59))
        ring_cont, hierarchy = cv.findContours(rings, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        kol = cv.drawContours(cv.cvtColor(rings.copy(), cv.COLOR_GRAY2BGR), ring_cont, -1, (0, 255, 255), 2)

        warped = self.warped

        blurred_cropped_hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)

        saturation_max = 255
        value_max = 170

        black_min = np.array([0, 0, 0], np.uint8)
        black_max = np.array([179, saturation_max, value_max], np.uint8)
        frame_threshed_black = cv.inRange(blurred_cropped_hsv, black_min, black_max)
        frame_threshed_black = cv.medianBlur(frame_threshed_black, 11)
        circles = cv.HoughCircles(frame_threshed_black, cv.HOUGH_GRADIENT, 1.5, 100)

        ring_cont, hierarchy = cv.findContours(frame_threshed_black, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # kol = cv.drawContours(cv.cvtColor(frame_threshed_black.copy(), cv.COLOR_GRAY2BGR), ring_cont, -1, (0, 255, 255), 2)
        kol = cv.cvtColor(frame_threshed_black.copy(), cv.COLOR_GRAY2BGR)
        sorted_conts = sorted(ring_cont, key=cv.contourArea)
        sorted_conts.reverse()
        # cv.drawContours(kol, sorted_conts[0], -1, (0, 0, 255),2)
        big_circle = (5,5,5)
        output = kol.copy()
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            # loop over the (x, y) coordinates and radius of the circles
            big_circle = max(circles, key=Image_Tools.circle_area)
            x, y, r = big_circle
            cv.circle(output, (x, y), r, (255, 0, 0), 3)
            # show the output image
            #GUI.im_compare([frame_threshed_black, output])
        x,y,r = big_circle
        cv.circle(kol, (x,y),r, (0, 0, 255), thickness=3)
        #GUI.imShow(kol)
        mask = np.zeros((1000, 1000), dtype='uint8')
        bg = np.zeros((1000, 1000, 3), dtype='uint8')
        bg[:] = (0, 0, 255)
        # mask[:] = (0, 0, 0)
        cv.circle(mask, (x,y),r, (255, 255, 255), thickness=-1)

        cyman = cv.bitwise_xor(mask, second)
        #GUI.imShow(cyman)
        # mask[:] = (255, 255, 255)
        cv.circle(mask, (x,y), r, (0, 0, 0), thickness=-1)
        circle = cv.bitwise_not(mask, bg)
        # GUI.imShow(cropped)

        return cyman

    def calc_cords(self, mean_lines, k):
        cords = []
        quadrant = 0
        if k < 6:
            quadrant = 0
        elif k < 10:
            quadrant = 1
        elif k < 16:
            quadrant = 2
        elif k < 20:
            quadrant = 3

        for line in mean_lines:
            y_cut = line[0]
            steep = line[1]
            y_check = steep * 250 + y_cut
            if (y_check < 500 and (quadrant == 0 or quadrant == 3)) or (
                    y_check > 500 and (quadrant == 1 or quadrant == 2)):
                cords.append([0, 500])
            else:
                cords.append([500, 1000])
        return cords

    def get_outer_ring_new(self):
        warped = self.warped

        blurred_cropped_hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV)

        saturation_max = 255
        value_max = 170

        black_min = np.array([0, 0, 0], np.uint8)
        black_max = np.array([179, saturation_max, value_max], np.uint8)
        frame_threshed_black = cv.inRange(blurred_cropped_hsv, black_min, black_max)
        frame_threshed_black = cv.medianBlur(frame_threshed_black, 11)
        circles = cv.HoughCircles(frame_threshed_black, cv.HOUGH_GRADIENT, 1.5, 100)


        ring_cont, hierarchy = cv.findContours(frame_threshed_black, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        #kol = cv.drawContours(cv.cvtColor(frame_threshed_black.copy(), cv.COLOR_GRAY2BGR), ring_cont, -1, (0, 255, 255), 2)
        kol = cv.cvtColor(frame_threshed_black.copy(), cv.COLOR_GRAY2BGR)
        sorted_conts = sorted(ring_cont, key=cv.contourArea)
        sorted_conts.reverse()
        #cv.drawContours(kol, sorted_conts[0], -1, (0, 0, 255),2)

        output = kol.copy()
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            # loop over the (x, y) coordinates and radius of the circles
            big_circle = max(circles, key=Image_Tools.circle_area)
            x,y,r = big_circle
            cv.circle(output,(x,y), r, (255,0,0),3)
            # show the output image
            GUI.im_compare([frame_threshed_black, output])

