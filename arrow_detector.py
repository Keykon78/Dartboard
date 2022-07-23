import cv2
import numpy as np
import sklearn
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

from gui import GUI
from image_tools import Image_Tools


# print(cv2.__version__)
class Arrow_Detector:
    ENV = {
        'DETECTION_KERNEL_SIZE': (100, 100),
        'DETECTION_RADIAL_STEP': 10,
        'DETECTION_KERNEL_THICKNESS': 1,
        'DETECTION_APEX_OFFSET': 20,  # 20
        'DETECTION_APEX_LINE_THICKNESS': 10,  # 20
        'DETECTION_APEX_LINE_THICKNESS_PEAK': 10,  # 20

        'APEX_CLIPPING_OFFSET': 50,
        'APEX_MARK_SIZE': 10
    }

    def detectArrowState(self, IM_arrow):
        lu = IM_arrow[0:IM_arrow.shape[0] // 2, 0:IM_arrow.shape[1] // 2]
        ru = IM_arrow[0:IM_arrow.shape[0] // 2, IM_arrow.shape[1] // 2:IM_arrow.shape[1]]
        lb = IM_arrow[IM_arrow.shape[0] // 2:IM_arrow.shape[0], 0:IM_arrow.shape[1] // 2]
        rb = IM_arrow[IM_arrow.shape[0] // 2:IM_arrow.shape[0], IM_arrow.shape[1] // 2:IM_arrow.shape[1]]
        verbs = [('l', 'u'), ('r', 'u'), ('l', 'b'), ('r', 'b')]
        stack = [lu, ru, lb, rb]
        max = -1
        maxIdx = 0
        for i in range(len(stack)):
            if np.sum(stack[i]) > max:
                max = np.sum(stack[i])
                maxIdx = i
        # print(verbs[maxIdx])
        return verbs[maxIdx]

    def computeArrowOrientation(self, IM, arange, kernel):
        max_contour_length = 0
        max_angle = 0
        max_contour = 0
        max_img = 0
        for i in arange:
            kernel_rot = Image_Tools.rotateImage(kernel, i)
            closed = cv2.morphologyEx(IM, cv2.MORPH_CLOSE, kernel_rot)
            closed = cv2.blur(closed, (11,11))
            contours, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for j in range(len(contours)):
                length = cv2.arcLength(contours[j], True)
                if length > max_contour_length:
                    max_contour_length = length
                    max_angle = i
                    max_contour = contours[j]
                    max_img = closed
        return max_contour_length, max_angle, max_contour, max_img

    def _detectArrowLine(self, IM_closed, max_contour, xx, yy, ww, hh):
        # Improve with fitting line
        line_image = np.zeros(IM_closed.shape, "uint8")
        line_image_peak = np.zeros(IM_closed.shape, "uint8")

        # then apply fitline() function
        [vx, vy, x, y] = cv2.fitLine(max_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        line = cv2.fitLine(max_contour, cv2.DIST_L2, 0, 0.01, 0.01)
        # Now find two extreme points on the line to draw line
        righty = int((-x * vy / vx) + y)
        lefty = int(((line_image.shape[1] - x) * vy / vx) + y)

        # Finally draw the line
        cv2.line(line_image, (line_image.shape[1] - 1, lefty), (0, righty), 255,
                 Arrow_Detector.ENV['DETECTION_APEX_LINE_THICKNESS'])
        cv2.line(line_image_peak, (line_image.shape[1] - 1, lefty), (0, righty), 255,
                 Arrow_Detector.ENV['DETECTION_APEX_LINE_THICKNESS_PEAK'])

        GUI.imShow_debug(line_image_peak)
        # compute orientation
        (h, v) = self.detectArrowState(Image_Tools.getROI(IM_closed, xx, yy, ww, hh))
        if h == 'l':
            if v == 'u':
                arrow_x1 = xx + ww
                arrow_y1 = yy + hh
            else:
                arrow_x1 = xx + ww
                arrow_y1 = yy
        else:
            if v == 'u':
                arrow_x1 = xx
                arrow_y1 = yy + hh
            else:
                arrow_x1 = xx
                arrow_y1 = yy
        return arrow_x1, arrow_y1, line_image_peak, h, v

    def _detectApex(self, IM_ROI2_grey, line_image_peak, arrow_x1, arrow_y1, h, v):
        # Isolate the apex
        offset = Arrow_Detector.ENV['DETECTION_APEX_OFFSET']
        IM_ROI_APEX = IM_ROI2_grey[arrow_y1 - offset:arrow_y1 + offset, arrow_x1 - offset:arrow_x1 + offset]
        IM_ROI_LINE = line_image_peak[arrow_y1 - offset:arrow_y1 + offset, arrow_x1 - offset:arrow_x1 + offset]
        IM_ROI_APEX_edges = cv2.Canny(IM_ROI_APEX, 50, 100)
        IM_ROI_APEX_masekd = cv2.multiply(IM_ROI_LINE, IM_ROI_APEX_edges)

        #GUI.imShow(IM_ROI_APEX)
        #GUI.imShow(IM_ROI_APEX_edges)

        contours_line, hierarchy_line = cv2.findContours(IM_ROI_APEX_masekd.copy(), cv2.RETR_LIST,
                                                         cv2.CHAIN_APPROX_SIMPLE)

        if len(contours_line) == 0:
            print("Errror")
            return None, None, None, None, None, None, None, None, None, None

        max_contour_idx = Image_Tools.getMaxContourIdx(contours_line)
        xxx, yyy, www, hhh = cv2.boundingRect(contours_line[max_contour_idx])

        # GUI.imShow(Image_Tools.debugRectangle(IM_ROI_APEX_masekd,xxx,yyy,www,hhh))
        # GUI.imShow(IM_ROI_APEX_masekd)

        IM_ROI_APEX_clipped = np.zeros(IM_ROI_APEX_masekd.shape, "uint8")
        IM_ROI_APEX_clipped[yyy:yyy + hhh, xxx:xxx + www] = IM_ROI_APEX_masekd[yyy:yyy + hhh, xxx:xxx + www]

        IM_ROI_APEX_masekd = IM_ROI_APEX_clipped
        # GUI.imShow(IM_ROI_APEX_clipped)

        # respect orientation
        y, x = np.where(IM_ROI_APEX_masekd > 1)
        np.sort(y)
        # print(h)
        # print(v)
        if h == 'l':
            if v == 'u':
                arrow_y2 = y[y.shape[0] - 1]
                x = np.where(IM_ROI_APEX_masekd[arrow_y2, :] > 1)[0]
                np.sort(x)
                arrow_x2 = x[x.shape[0] - 1]
            else:
                arrow_y2 = y[0]
                x = np.where(IM_ROI_APEX_masekd[arrow_y2, :] > 1)[0]
                arrow_x2 = x[x.shape[0] - 1]
        else:
            if v == 'u':
                arrow_y2 = y[y.shape[0] - 1]
                x = np.where(IM_ROI_APEX_masekd[arrow_y2, :] > 1)[0]
                np.sort(x)
                arrow_x2 = x[0]
                # arrow_y2 = yyy
                # arrow_x2 = xxx
            else:
                arrow_y2 = y[0]
                x = np.where(IM_ROI_APEX_masekd[arrow_y2, :] > 1)[0]
                np.sort(x)
                arrow_x2 = x[0]

                # transform to original space
        arrow_y1 = (arrow_y1 - offset) + arrow_y2
        arrow_x1 = (arrow_x1 - offset) + arrow_x2

        return arrow_x1, arrow_y1, IM_ROI_APEX

    def detect_roi(self, diff_image ):

        white_px = cv2.findNonZero(diff_image)
        blp = cv2.cvtColor(diff_image.copy(), cv2.COLOR_GRAY2BGR)
        clean_lst = [list(px[0]) for px in white_px]
        x_vals = [px[0] for px in  clean_lst]
        y_vals = [px[1] for px in clean_lst]

        cluster = AgglomerativeClustering(distance_threshold=30.0, affinity='euclidean', linkage='single', n_clusters=None)
        cluster.fit_predict(clean_lst)

        # plt.scatter(x_vals, y_vals, c=cluster.labels_, cmap='rainbow')
        # plt.show()
        for i,px in enumerate(clean_lst):
            if cluster.labels_[i] == 0:
                blp[px[1]][px[0]] = (0,0,255)
            if cluster.labels_[i] == 1:
                blp[px[1]][px[0]] = (255,255,0)
            if cluster.labels_[i] == 2:
                blp[px[1]][px[0]] = (0,255,0)
            if cluster.labels_[i] == 3:
                blp[px[1]][px[0]] = (255,0,0)

        label_list = list(cluster.labels_)
        combined_lists = [[] for i in range(max(label_list)+1)]
        combined_lists = [[] for i in range(max(label_list)+1)]

        for k,px in enumerate(clean_lst):
            combined_lists[label_list[k]].append(px)
        max_index = combined_lists.index(max(combined_lists, key=len))
        for px in combined_lists[max_index]:
            blp[px[1]][px[0]] = (255, 0, 255)
        GUI.imShow_debug(blp)
        return combined_lists[max_index]

    def detectArrow(self, diff_image, warp_mode, warped_frame):
        kernel_size = Arrow_Detector.ENV['DETECTION_KERNEL_SIZE']
        kernel = np.zeros(kernel_size, np.uint8)
        kernel_thickness = Arrow_Detector.ENV['DETECTION_KERNEL_THICKNESS']
        kernel[:, (kernel.shape[1] // 2) - kernel_thickness:(kernel.shape[1] // 2) + kernel_thickness] = 1
        roi = self.detect_roi(diff_image)
        roi_can = np.zeros((diff_image.shape), dtype='uint8')
        for px in roi:
            roi_can[px[1], px[0]] = 255
        GUI.imShow_debug(roi_can)
        max_contour_length, max_angle, max_contour, max_img = self.computeArrowOrientation(roi_can, range(0, 180,
                                                                                                             Arrow_Detector.ENV[
                                                                                                                     'DETECTION_RADIAL_STEP']),
                                                                                           kernel)
        blp = cv2.cvtColor(diff_image.copy(), cv2.COLOR_GRAY2BGR)
        #cv2.drawContours(blp, max_contour, -1, (0,0,255), 10)
        #GUI.imShow(blp)


        #GUI.imShow(roi_can)
        if len(max_contour) == 0:
            return None, None, None, None, None, None, None, None, None, None

        xx, yy, ww, hh = cv2.boundingRect(max_contour)
        xx, yy, ww, hh = cv2.boundingRect(roi_can)
        arrow_y1=yy
        arrow_x1 = 999
        width = 1
        tol = width
        tol_broken = False
        peak_width = 0
        end_x = 0
        starting_point = yy
        end_point = yy + hh
        y_cords = list(range(starting_point, end_point))



        if warp_mode == "down":
            y_cords = list(reversed(y_cords))
        elif warp_mode == "up":
            y_cords = y_cords
        else:
            print("no warpmode")
            return

        arrow_x1 = end_x -int(peak_width/2)
        cv2.rectangle(blp, (xx, yy), (xx + ww, yy + hh), (0, 0, 255))

        [vx, vy, x, y] = cv2.fitLine(max_contour, cv2.DIST_L2, 0, 0.01, 0.01)

        righty = int((-x * vy / vx) + y)
        lefty = int(((blp.shape[1] - x) * vy / vx) + y)
        pt1 = (blp.shape[1] - 1, lefty)
        pt2 = (0, righty)
        steep = (pt1[1]-pt2[1])/(pt1[0]-pt2[0])
        y_cut = pt1[1] - (pt1[0] * steep)

        # Finally draw the line
        # cv2.line(blp, pt1, pt2, 255,
        #          Arrow_Detector.ENV['DETECTION_APEX_LINE_THICKNESS'])
        cv2.putText(blp, "Neue Methode", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2,
                   cv2.LINE_AA)
        cv2.line(blp, pt1, pt2, 255,
                 Arrow_Detector.ENV['DETECTION_APEX_LINE_THICKNESS_PEAK'])
        arrow_y1 = int(y_cords[1])
        arrow_x1 = int((arrow_y1 -y_cut)/steep)

        cv2.circle(blp, (arrow_x1, arrow_y1), 4, (0, 0, 255), -1)
        #print("a" + str(arrow_x1) + " "  +" " + str(arrow_y1))

        IM_ROI_grey = self.markApex(warped_frame, arrow_x1, arrow_y1)
        GUI.im_compare([blp,IM_ROI_grey])
        return arrow_x1, arrow_y1

    def debugApex(self, IM, arrow_x, arrow_y, color):
        IM = IM.copy()
        clipping_offset = Arrow_Detector.ENV['APEX_CLIPPING_OFFSET']
        cv2.rectangle(IM, (arrow_x, arrow_y), (arrow_x, arrow_y), color, 2)
        IM_arrow_roi = IM[arrow_y - clipping_offset:arrow_y + clipping_offset,
                       arrow_x - clipping_offset:arrow_x + clipping_offset]
        # show(IM_arrow_roi)
        return IM_arrow_roi

    def markApex(self, IM_ROI, arrow_x, arrow_y):
        IM_ROI = cv2.cvtColor(IM_ROI, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(IM_ROI, (arrow_x, arrow_y), (arrow_x, arrow_y), (0, 0, 255),
                      Arrow_Detector.ENV["APEX_MARK_SIZE"])
        return IM_ROI

    def getMetricOfArrow(self, IM_ROI_ROTATED):
        ret2, thred = cv2.threshold(IM_ROI_ROTATED, 254, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thred.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # if len(contours) == 0:
        #     return None, None, None, None, None, None, None

        IM_dot = thred.copy()

        # IM_dot = thred.copy()
        # cnt = contours[0]
        # MO = cv2.moments(cnt)
        # cx = int(MO['m10'] / MO['m00'])
        # cy = int(MO['m01'] / MO['m00'])
        # cv2.line(thred, (thred.shape[0] // 2, thred.shape[1] // 2), (cx, cy), (255, 255, 255), 2)
        # IM_line = thred
        # dx = cx - (thred.shape[0] // 2)
        # dy = cy - (thred.shape[1] // 2)
        # length = np.sqrt((dx * dx) + (dy * dy))
        # nx = 0
        # ny = -1
        # angle = np.arccos(((nx * dx) + (ny * dy)) / length)
        # cross = (dx * ny) - (nx * dy)
        # if cross > 0:
        #     angle = (np.pi * 2) - angle

        #angle = np.rad2deg(angle)
        #return cx, cy, angle, length, cross, IM_dot, IM_line
        return IM_dot
