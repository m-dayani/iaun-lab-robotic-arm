import os
import numpy as np
import cv2
import sys
import collections
from scipy.spatial.transform import Rotation as scipyR

sys.path.append('./calib')
from cameraCalibration import recalib_camera

from hough_transform import find_circles

# from aun_robo_arm import LabCamera


# First version of square detector
# 1. calculate corners
# 2. optionally calculate the ROI by template matching
# 3. sort by response
# 4. perform a kind of non-max suppression:
#   a) pick the best response
#   b) remove all other points with a dist less than a th
#   c) group items in b) under the best response to average
#   d) when all points are classified, look for the structure: which points make a rectangle
#   e) can optionally refine the results by ROI

def point_dist(pt1, pt2):
    return np.sqrt(np.sum(np.square(pt1 - pt2)))


def calc_object_center(patch):
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(patch, ddepth, 1, 0)
    dy = cv2.Sobel(patch, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)

    th = 100
    mask = mag > th
    w, h = mask.shape

    X = np.repeat(np.arange(0, h).reshape((1, h)), w, axis=0)
    Y = np.repeat(np.arange(0, w).reshape((w, 1)), h, axis=1)
    x_sel = X[mask]
    y_sel = Y[mask]
    obj_x = np.sum(x_sel) / len(x_sel)
    obj_y = np.sum(y_sel) / len(y_sel)

    return np.array([obj_x, obj_y])


def detect_black_square(image, patch):
    # Template Matching (find ROI)
    tm_res = cv2.matchTemplate(image, patch, cv2.TM_SQDIFF_NORMED)
    _, _, tm_loc, _ = cv2.minMaxLoc(tm_res)

    pt1 = np.array([tm_loc[0], tm_loc[1]])
    pt2 = pt1 + np.array([patch.shape[1], patch.shape[0]])

    # mask = np.zeros(image.shape)
    # mask[pt1[1]:pt2[1], pt1[0]:pt2[0]] = 1.0

    # Detect Harris Corners
    gray = np.float32(image)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    scores = dict()
    for r in range(pt1[1], pt2[1]):
        for c in range(pt1[0], pt2[0]):
            scores[dst[r, c]] = (r, c)

    l = list(scores.items())
    l.sort(reverse=True)  # sort in reverse order
    scores = dict(l)

    th_res = 0.1
    th_px = 2
    best_res = -1
    idx = 0
    res = dict()
    for score in scores.keys():

        if best_res < score:
            best_res = score
        elif score < th_res * best_res:
            break

        loc = np.array(scores[score])

        matched_idx = -1
        for pt in res.keys():
            pt_loc = res[pt]['avg']
            d = point_dist(loc, pt_loc)

            if d < th_px:
                matched_idx = pt
                break

        if matched_idx >= 0:
            # append point to list
            curr_n = res[matched_idx]['n']
            res[matched_idx]['loc'].append(loc)
            res[matched_idx]['avg'] = (res[matched_idx]['avg'] * curr_n + loc) / (curr_n + 1)
            res[matched_idx]['n'] = len(res[matched_idx]['loc'])
        else:
            # create new group
            res[idx] = dict()
            res[idx]['loc'] = []
            res[idx]['loc'].append(loc)
            res[idx]['n'] = len(res[idx]['loc'])
            res[idx]['avg'] = loc
            idx += 1

    points = []
    for key in res.keys():
        points.append(res[key]['avg'])
        if len(points) >= 4:
            break
    points = np.array(points)

    # Structural Analysis
    if len(points) < 4:
        return []

    # which point is the origin, x-axis, and y-axis
    # determine the x-axis
    x_idx = 0
    dx_arr = [abs(points[:, 1] - 0), abs(points[:, 1] - image.shape[1])]
    if min(dx_arr[1]) < min(dx_arr[0]):
        x_idx = 1
    y_idx = 0
    dy_arr = [abs(points[:, 0] - 0), abs(points[:, 0] - image.shape[0])]
    if min(dy_arr[1]) < min(dy_arr[0]):
        y_idx = 1

    orig_idx = np.argmin(dx_arr[x_idx] * dy_arr[y_idx])

    dx_orig = abs(points[:, 1] - points[orig_idx, 1])
    dx_orig[orig_idx] += max(dx_orig)
    y_idx = np.argmin(dx_orig)

    dy_orig = abs(points[:, 0] - points[orig_idx, 0])
    dy_orig[orig_idx] += max(dy_orig)
    x_idx = np.argmin(dy_orig)

    xy_idx = list(set(range(0, 4)) - {orig_idx, x_idx, y_idx})[0]

    ord_pts = np.array([points[orig_idx], points[x_idx], points[y_idx], points[xy_idx]])

    # If this is a square, sides must be almost identical
    th_square = 0.4
    d0 = point_dist(ord_pts[0], ord_pts[1])
    d1 = point_dist(ord_pts[0], ord_pts[2])
    d2 = point_dist(ord_pts[3], ord_pts[1])
    d3 = point_dist(ord_pts[3], ord_pts[2])

    c1 = abs(d0 / d1 - 1.0) < th_square
    c2 = abs(d1 / d2 - 1.0) < th_square
    c3 = abs(d2 / d3 - 1.0) < th_square

    if c1 and c2 and c3:
        return ord_pts
    else:
        return []


# todo: generalize the above idea to more complex patterns

def calib_square(points, K):
    K_1 = np.linalg.inv(K)

    if len(points) <= 0:
        return

    # Note the direction and order of points in pixels (rows, cols) and in image (x-y)
    dx = points[1] - points[0]
    dx_vec = np.array([dx[1], dx[0], 1.0]).reshape((3, 1))
    rx = K_1 @ dx_vec
    s = 5.0 / np.linalg.norm(rx)
    rx = s * rx / 5.0

    dy = points[2] - points[0]
    dy_vec = np.array([dy[1], dy[0], 1.0]).reshape((3, 1))
    ry = (s / 5.0) * K_1 @ dy_vec

    rz = np.cross(rx.reshape((1, 3)), ry.reshape((1, 3))).reshape((3, 1))

    R_cw = np.concatenate([rx, ry, rz], axis=1)
    rr = scipyR.from_matrix(R_cw)
    print(rr.as_rotvec())

    true_r = np.array([[-0.27446404],
                       [-0.10632083],
                       [-2.95396537]])
    print(true_r)

    t_cw = s * K_1 @ np.array([points[0, 1], points[0, 0], 1.0]).reshape((3, 1))


def smooth(img, gk_size=(3, 3), mk_size=5):
    # make sure images are grayscale
    gray = img
    if len(gray.shape) >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # perform gaussian denoising followed by median filtering
    gauss = cv2.GaussianBlur(gray, gk_size, cv2.BORDER_DEFAULT)
    med = cv2.medianBlur(gauss, mk_size)

    return med


def get_mesh(size):
    xsz = size[0]
    ysz = size[1]
    x = np.arange(ysz)
    y = np.arange(xsz)
    X = np.repeat(x.reshape((1, len(x))), xsz, axis=0)
    Y = np.repeat(y.reshape((len(y), 1)), ysz, axis=1)

    return X, Y


def find_roi_thresh(th_img):
    mask = th_img > 0
    img_size = th_img.shape[:2]
    x_min = 0
    x_max = img_size[0]
    y_min = 0
    y_max = img_size[1]

    if not mask.any():
        return np.array([x_min, x_max, y_min, y_max])

    X, Y = get_mesh(img_size)
    pt_x = X[mask]
    pt_y = Y[mask]

    if len(pt_x) > 0:
        x_min = pt_x[np.argmin(pt_x)]
        x_max = pt_x[np.argmax(pt_x)]
        y_min = pt_y[np.argmin(pt_y)]
        y_max = pt_y[np.argmax(pt_y)]

    return np.array([x_min, x_max, y_min, y_max])


def track_hough_circles(img, last_img):
    # smooth both images
    img_s = smooth(img)
    last_img_s = smooth(last_img)

    # subtract images to find ROI
    sub_img = cv2.addWeighted(img_s, 0.5, last_img_s, -0.5, 255)
    # cv2.normalize(sub_img, sub_img, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(sub_img, None)
    ret, thresh = cv2.threshold(sub_img, 230, 255, cv2.THRESH_BINARY_INV)

    # find ROI
    roi = find_roi_thresh(thresh)
    center_roi = [(roi[0] + roi[1]) * 0.5, (roi[2] + roi[3]) * 0.5]

    # find circles
    circles = find_circles(img_s, hparams=(8, 60, 30, 10, 70))

    if circles is not None:
        if circles.shape[1] != 1:
            circles = circles.squeeze()
        else:
            circles = circles[0]
        dist_circles = []
        for circle in circles:
            dist_circles.append(point_dist(center_roi, circle[:2]))
        min_dist = np.argmin(dist_circles)

        return circles[min_dist]
    else:
        return np.array([center_roi[1], center_roi[0], 10])


def test_squares(base_dir):

    file_name = os.path.join(base_dir, 'calib', 'images', 'image0.png')
    patch_name = os.path.join(base_dir, 'obj_detection', 'images', 'ball_patch.png')

    # load image
    im_color = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(im_color, cv2.COLOR_RGB2GRAY)
    # load patch
    patch = cv2.imread(patch_name, cv2.IMREAD_GRAYSCALE)

    points = detect_black_square(image, patch)

    K = np.array([[625.24678685, 0., 297.50306391],
                  [0., 624.90293937, 251.99629151],
                  [0., 0., 1.]])
    calib_square(points, K)

    im_show = im_color
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)]
    for i in range(len(points)):
        im_show = cv2.circle(im_show, (int(points[i, 1]), int(points[i, 0])), 3, colors[i])

    # cv2.imshow('Image', im_show)
    # # cv2.imwrite(os.path.join(base_dir, 'patch0.png'), image[10:40, 540:580])
    #
    # cv2.waitKey(0)

    obj_center = calc_object_center(patch)

    im_show = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    im_show = cv2.drawMarker(im_show, np.int32(obj_center), (0, 0, 255), cv2.MARKER_CROSS, 3, 3)

    cv2.imshow('Image', im_show)
    cv2.waitKey(0)


def test_hough_circles(base_dir):

    video_file = os.path.join(base_dir, 'obj_detection', 'images', 'output.avi')
    another_video = os.path.join(base_dir, '..', '20231103_084334.mp4')
    scale = 0.5
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error opening video file")

    last_frame = []

    while cap.isOpened():
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        if ret:
            if len(last_frame) <= 0:
                last_frame = frame
                sub_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                circle = track_hough_circles(frame, last_frame)
                last_frame = frame

                circle = np.int32(circle)
                frame = cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


class TrackerKLT:

    def __init__(self):
        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.p0 = None
        self.p1 = None
        self.old_gray = None
        self.is_initialized = False

    def initialize(self, frame, mask=None):
        self.p0 = cv2.goodFeaturesToTrack(frame, mask=mask, **self.feature_params)
        self.old_gray = frame.copy()
        self.is_initialized = True

    def track(self, frame):
        self.p1, self.st, self.err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame, np.float32(self.p0),
                                                              None, **self.lk_params)
        self.old_gray = frame.copy()
        self.p0 = self.p1

    def updatePoints(self, pt):
        self.p0 = pt


def test_klt_of(base_dir):

    video_file = os.path.join(base_dir, '..', 'slow_traffic_small.mp4')
    video_file1 = os.path.join(base_dir, 'obj_detection', 'images', 'output.avi')
    video_file2 = os.path.join(base_dir, '..', '20231103_084334.mp4')

    cap = cv2.VideoCapture(video_file1)

    if not cap.isOpened():
        print("Error opening video file")
        return

    kltTracker = TrackerKLT()

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    kltTracker.initialize(old_gray)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kltTracker.track(frame_gray)

        # Select good points
        if kltTracker.p1 is not None:
            st = kltTracker.st
            good_new = kltTracker.p1[st == 1]
            good_old = kltTracker.p0[st == 1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update previous points
        kltTracker.updatePoints(good_new.reshape(-1, 1, 2))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    base_dir = os.getenv('DATA_PATH')
    # test_squares(base_dir)
    # test_hough_circles(base_dir)
    test_klt_of(base_dir)

