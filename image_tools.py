import math

import cv2
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt

from gui import GUI


class Image_Tools:

    @staticmethod
    def getMaxContourIdx(contours):
        max_contour_length = 0
        max_contour = None
        for j in range(len(contours)):
            length = cv2.arcLength(contours[j], True)
            if length > max_contour_length:
                max_contour_length = length
                max_contour = j
        return max_contour

    @staticmethod
    def readFrame(time_in_millis, size):
        cap.set(cv2.CAP_PROP_POS_MSEC, time_in_millis)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, size)
            return frame

    @staticmethod
    def readAndSafeFrames(path):
        cap = cv2.VideoCapture(path)
        frames = []
        frames.append(readFrame(100))
        frames.append(readFrame(2000))
        frames.append(readFrame(3000))
        frames.append(readFrame(5000))
        cap.release()
        for i in range(len(frames)):
            cv2.imwrite("Vid" + str(i) + ".png", frames[i])

    @staticmethod
    def normalizeHist(arr):
        minval = arr[:, :].min()
        maxval = arr[:, :].max()
        print(minval)
        print(maxval)
        if minval != maxval:
            arr -= minval
            arr *= int((255.0 / (maxval - minval)))
        return arr

    @staticmethod
    def rotateImage(image, angle):
        image_center = tuple(np.array(image.shape) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape, flags=cv2.INTER_LINEAR)
        return result

    @staticmethod
    def sift(GrayIM):
        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(GrayIM, None)
        out = np.array(GrayIM.shape)
        out = cv2.drawKeypoints(GrayIM, kp, out)
        return out, kp

    @staticmethod
    def debugRectangle(IM, x, y, w, h):
        IM_copy = IM.copy()
        cv2.rectangle(IM_copy, (x, y), (x + w, y + h), (255, 255, 255), 1)
        return IM_copy

    @staticmethod
    def debugContours(IM, contours):
        SAMPLE = np.zeros(IM.shape, "uint8")
        cv2.drawContours(SAMPLE, contours, -1, (255, 255, 255), 10)
        return SAMPLE

    @staticmethod
    def getROI(IM, x, y, w, h):
        if x < 0:
            x = 0
        if (IM.ndim == 2):
            IM_ROI = IM[y:y + h, x:x + w]
        else:
            IM_ROI = IM[y:y + h, x:x + w, :]
        return IM_ROI

    @staticmethod
    def readImage(path, dimension=None):
        IM = cv2.imread(path)
        if dimension is not None:
            IM = cv2.resize(IM, dimension)
        if IM.ndim == 3:
            base_frame_gray = cv2.cvtColor(IM, cv2.COLOR_BGR2GRAY)
        # print(IM.shape)
        # print(IM.dtype)
        # show(IM)
        return IM, base_frame_gray

    @staticmethod
    def prepareImage(IM, dimension):
        IM = cv2.resize(IM, dimension)
        base_frame_gray = cv2.cvtColor(IM, cv2.COLOR_BGR2GRAY)
        # print(IM.shape)
        # print(IM.dtype)
        # show(IM)
        return IM, base_frame_gray

    @staticmethod
    def getIntersection(src_points):
        if src_points.shape != (4, 2):
            return None, None
        # interesect lines
        x1 = src_points[0, 0]
        y1 = src_points[0, 1]
        x2 = src_points[3, 0]
        y2 = src_points[3, 1]
        x3 = src_points[1, 0]
        y3 = src_points[1, 1]
        x4 = src_points[2, 0]
        y4 = src_points[2, 1]
        py = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        px = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
                (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        # show(IM)
        # swap output
        return int(px), int(py)

    @staticmethod
    def debugIntersection(IM, src_points):
        IM = IM.copy()
        x1 = src_points[0, 0]
        y1 = src_points[0, 1]
        x2 = src_points[3, 0]
        y2 = src_points[3, 1]
        x3 = src_points[1, 0]
        y3 = src_points[1, 1]
        x4 = src_points[2, 0]
        y4 = src_points[2, 1]
        cv2.line(IM, (int(x1), int(y1)), (int(x2), int(y2)), 255, 5)
        cv2.line(IM, (int(x3), int(y3)), (int(x4), int(y4)), 255, 5)
        return IM

    @staticmethod
    def scale_contour(cnt, scale):
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cnt_norm = cnt - [cx, cy]
        cnt_scaled = cnt_norm * scale
        cnt_scaled = cnt_scaled + [cx, cy]
        cnt_scaled = cnt_scaled.astype(np.int32)

        return cnt_scaled

    @staticmethod
    def low_cut(contours, groups, ret_list=False):
        areas = [[cv2.contourArea(cont), i] for i, cont in enumerate(contours)]

        kmeans = KMeans(n_clusters=groups, random_state=0).fit(areas)
        categories = kmeans.labels_
        raw = [cv2.contourArea(cont) for cont in contours]
        # plt.scatter(raw, list(range(len(raw))), c=categories, cmap='rainbow')
        # plt.show()
        combined_area = list(np.swapaxes([list(categories), raw], 0, 1))
        area_in_cat = []
        cont_in_cat = []
        for k in range(groups):
            area_in_cat.append([])
            cont_in_cat.append([])
        for i, areas in enumerate(combined_area):
            area_in_cat[int(areas[0])].append(areas[1])
            cont_in_cat[int(areas[0])].append(contours[i])

        maxis = [max(cat) for cat in area_in_cat]
        min_max = min(maxis)
        banish = -1
        for i, val in enumerate(area_in_cat):
            if min_max in val:
                banish = i
        cont_in_cat.remove(cont_in_cat[banish])
        filtered_cons = []
        for cont in cont_in_cat:
            filtered_cons += cont
        cont_in_cat = sorted(cont_in_cat, key=Image_Tools.take_area_of_max)
        return filtered_cons, cont_in_cat

    @staticmethod
    def low_cut_unsupervised(contours):
        cluster = AgglomerativeClustering(distance_threshold=30.0, affinity='euclidean', linkage='single',
                                          n_clusters=None)
        raw_x = [cv2.contourArea(contour) for contour in contours]
        combined = [[cv2.contourArea(contour), i * 0.01] for i, contour in enumerate(contours)]
        cluster.fit_predict(combined)
        # plt.scatter(raw_x, list(range(len(raw_x))), c=cluster.labels_, cmap='rainbow')
        # plt.show()
        label_list = list(cluster.labels_)

        area_in_cat = [[] for i in range(max(label_list) + 1)]
        cont_in_cat = [[] for i in range(max(label_list) + 1)]

        for k, contour in enumerate(contours):
            area_in_cat[label_list[k]].append(cv2.contourArea(contour))
            cont_in_cat[label_list[k]].append(contour)

        maxis = [max(cat) for cat in area_in_cat]
        min_max = min(maxis)
        banish = -1
        for i, val in enumerate(area_in_cat):
            if min_max in val:
                banish = i
        cont_in_cat.remove(cont_in_cat[banish])
        filtered_cons = []
        for cont in cont_in_cat:
            filtered_cons += cont
        cont_in_cat = sorted(cont_in_cat, key=Image_Tools.take_area_of_max)
        return filtered_cons, cont_in_cat

    @staticmethod
    def take_area_of_max(element):
        sort_max_area = max(element, key=cv2.contourArea)
        return cv2.contourArea(sort_max_area)

    @staticmethod
    def create_noise(img, size=4, number=5):
        cop = np.zeros((img.shape[1], img.shape[0], 1), dtype='uint8')
        for i in range(number):
            cx = np.random.randint(cop.shape[1])
            cy = np.random.randint(cop.shape[0])
            cv2.circle(cop, (cx, cy), size, (255), -1)

        noise, hierachy = cv2.findContours(cop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return noise

    @staticmethod
    def get_contour_center(contour):
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return cx, cy

    @staticmethod
    def calc_roi(green_mask, red_mask):
        rings = cv2.blur(cv2.bitwise_or(green_mask, red_mask), (59, 59))
        ring_cont, hierarchy = cv2.findContours(rings, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # kol = cv.drawContours(rings, ring_cont, -1, (0, 255, 255), 2)
        # GUI.imShow(kol)
        big_contour = max(ring_cont, key=cv2.contourArea)
        big_contour = Image_Tools.scale_contour(big_contour, 1)

        xx, yy, ww, hh = cv2.boundingRect(big_contour)

        return xx, yy, ww, hh

    @staticmethod
    def zoom_roi(img, x, y, w, h):
        canvas = img
        cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 5)
        # GUI.imShow(canvas)
        return img[y:y + h, x:x + w]

    @staticmethod
    def crop_ellipse(img, ellipse):
        if img.ndim != 3:
            img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

        mask = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
        bg = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
        bg[:] = (0, 0, 255)
        mask[:] = (0, 0, 0)
        cv2.ellipse(mask, ellipse, (255, 255, 255), thickness=-1)
        cropped = cv2.bitwise_and(mask, img.copy())
        circle = cv2.bitwise_not(mask, bg)
        # GUI.imShow(circle)
        cropped = cv2.bitwise_or(circle, cropped)

        mask[:] = (255, 255, 255)
        cv2.ellipse(mask, ellipse, (0, 0, 0), thickness=-1)
        circle = cv2.bitwise_not(mask, bg)
        cropped_bg_black = cv2.bitwise_and(circle, cropped)
        # GUI.imShow(cropped)

        return cropped, cropped_bg_black

    @staticmethod
    def arclen(elem):
        return cv2.arcLength(elem, False)

    @staticmethod
    def circle_area(elem):
        x, y, r = elem
        area = math.pi * r ** 2
        return area

    @staticmethod
    def white_balance(img):
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result

    @staticmethod
    def get_brightness(img):

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        hsv_planes = cv2.split(hsv_img)
        # hist_size = 256
        # hist_range = (0, 256)
        # accumulate = False
        #
        # v_hist = cv2.calcHist(hsv_planes, [0], None, [hist_size], hist_range, accumulate=accumulate)
        #
        # GUI.imShow(hsv_planes[0])
        #
        # plt.plot(v_hist)
        # plt.show()

        return np.mean(hsv_planes[0])
