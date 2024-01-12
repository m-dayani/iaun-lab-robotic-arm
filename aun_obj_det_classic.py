import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as scipyR

from hough_transform import find_circles
from aun_imp_basics import img_resize, point_dist, get_mesh, smooth, calc_object_center


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


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def find_squares(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            contours, _hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4], cnt[(i + 2) % 4]) for i in range(4)])
                    if max_cos < 0.104:
                        squares.append(cnt)
    return squares


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

    return np.array([x_min, y_min, x_max, y_max])


# for each of the contours detected, the shape of the contours is approximated using approxPolyDP()
# function and the contours are drawn in the image using drawContours() function
def detect_objects(contours, blank):
    font_scale = 0.5
    f_thick = 1
    for count in contours:
        epsilon = 0.01 * cv2.arcLength(count, True)
        approximations = cv2.approxPolyDP(count, epsilon, True)
        cv2.drawContours(blank, [approximations], 0, (0, 255, 0), 3)
        # the name of the detected shapes are written on the image
        i, j = approximations[0][0]
        if len(approximations) == 3:
            cv2.putText(blank, "Triangle", (i, j), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), f_thick)
        elif len(approximations) == 4:
            cv2.putText(blank, "Rectangle", (i, j), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), f_thick)
        elif len(approximations) == 5:
            cv2.putText(blank, "Pentagon", (i, j), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), f_thick)
        elif 6 < len(approximations) < 15:
            cv2.putText(blank, "Ellipse", (i, j), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), f_thick)
        else:
            cv2.putText(blank, "Circle", (i, j), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 255), f_thick)
        # displaying the resulting image as the output on the screen
        cv2.imshow("Resulting_image", blank)
        cv2.waitKey(0)


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
    center_roi = [(roi[0] + roi[2]) * 0.5, (roi[1] + roi[3]) * 0.5]

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


def test_watershed(img):
    img = img_resize(img, 0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # cv.imshow('Thresh', thresh)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # cv.imshow("Sure_fg", unknown)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    cv2.imshow('final', img)

    cv2.waitKey(0)


if __name__ == "__main__":

    base_dir = os.getenv('DATA_PATH')
    # test_squares(base_dir)
    # test_hough_circles(base_dir)
    test_klt_of(base_dir)

