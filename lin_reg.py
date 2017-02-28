import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import math
import imageio
from moviepy.editor import VideoFileClip
from cachetools import LRUCache
from numpy.polynomial import Polynomial as polyeval

# download ffmpeg plugin
imageio.plugins.ffmpeg.download()

# setup image
image = cv2.imread('test_images/solidWhiteRight.jpg')

# setup cachetools
cache = LRUCache(maxsize=4)
slope_cache = LRUCache(maxsize=2)


def bgr_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def original_draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):

    right_lane_x = []
    left_lane_x = []
    right_lane_y = []
    left_lane_y = []

    right_lengths = []
    left_lengths = []
    right_slopes = []
    left_slopes = []

    for line in lines:
        for x1,y1,x2,y2 in line:

            # 1. get the slope and length
            slope = ((y2-y1)/(x2-x1))
            length = math.sqrt((x1-x2)**2 + (y1-y2)**2)

            # 2. run checks
            if 0.4 > slope > -0.4:
                continue

            if slope > 0.9:
                continue

            if slope < -0.9:
                continue

            # 3. separate them between positive and negative slope
            if slope > 0:
                # collect right lane x and y
                right_lane_x.extend([x1, x2])
                right_lane_y.extend([y1, y2])
                right_slopes.append(slope)
                right_lengths.append(length)

            else:
                # collect left lane x and y
                left_lane_x.extend([x1, x2])
                left_lane_y.extend([y1, y2])
                left_slopes.append(slope)
                left_lengths.append(length)

    # set y
    y1 = 540
    y2 = 335

    if right_lane_x and left_lane_x:

        # get weighted slope
        right_slope = get_weighted_slope(right_slopes, right_lengths)
        left_slope = get_weighted_slope(left_slopes, left_lengths)

        if slope_cache.currsize > 0:
            previous_left_slope = slope_cache["left_slope"]
            previous_right_slope = slope_cache["right_slope"]

            right_change_percent = ((right_slope - previous_right_slope)/previous_right_slope)*100.0
            left_change_percent = ((left_slope - previous_left_slope)/previous_left_slope)*100.0

            slope_cache["left_slope"] = left_slope
            slope_cache["right_slope"] = right_slope

            right_change_percent = abs(right_change_percent)
            left_change_percent = abs(left_change_percent)

            if right_change_percent > 1.5 or left_change_percent > 1.5:
                show_previous_line(img, y1, y2)

            else:
                print(left_change_percent)
                # get polynomial fit
                right_poly = polyeval.fit(np.array(right_lane_x), np.array(right_lane_y), 1)
                left_poly = polyeval.fit(np.array(left_lane_x), np.array(left_lane_y), 1)

                left_x1 = round((left_poly - y1).roots()[0])
                right_x1 = round((right_poly - y1).roots()[0])

                left_x2 = round((left_poly - y2).roots()[0])
                right_x2 = round((right_poly - y2).roots()[0])

                if cache.currsize > 0:
                    previous_left_x1 = cache["left_x1"]
                    previous_right_x1 = cache["right_x1"]
                    previous_left_x2 = cache["left_x2"]
                    previous_right_x2 = cache["right_x2"]

                    left_x1 = np.average([left_x1, previous_left_x1])
                    left_x2 = np.average([left_x2, previous_left_x2])
                    right_x1 = np.average([right_x1, previous_right_x1])
                    right_x2 = np.average([right_x2, previous_right_x2])

                cache["left_x1"] = left_x1
                cache["right_x1"] = right_x1
                cache["left_x2"] = left_x2
                cache["right_x2"] = right_x2

                cv2.line(img, (int(left_x1), int(y1)), (int(left_x2), int(y2)), color, thickness)
                cv2.line(img, (int(right_x1), int(y1)), (int(right_x2), int(y2)), color, thickness)
        else:
            slope_cache["left_slope"] = left_slope
            slope_cache["right_slope"] = right_slope

    else:
        show_previous_line(img, y1, y2)


def get_top_percentile(points):
    top_limit = np.percentile(points, 80)
    bottom_limit = np.percentile(points, 20)
    new_points = []
    mean_point = np.mean(points)
    for point in points:
        if point > top_limit and point > bottom_limit:
            new_points.append(point)
        else:
            new_points.append(mean_point)

    return new_points

def show_previous_line(img, y1, y2):
    if cache.currsize > 1:
        left_x1 = cache["left_x1"]
        right_x1 = cache["right_x1"]
        left_x2 = cache["left_x2"]
        right_x2 = cache["right_x2"]

        cv2.line(img, (int(left_x1), int(y1)), (int(left_x2), int(y2)), color=[255, 0, 0], thickness=10)
        cv2.line(img, (int(right_x1), int(y1)), (int(right_x2), int(y2)), color=[255, 0, 0], thickness=10)

def get_weighted_slope(slopes, lengths):
    weighted_slope = sum(slopes[g] * lengths[g] for g in range(len(slopes))) / sum(lengths)
    return weighted_slope

def get_y_intercept(m, x1, y1):
    b=y1-m*x1
    return b

def average_slope_intercept(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y)[0]
    return b

def get_mid_point(points):

    # TO DO: add weight here in terms of length

    return np.mean(points)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def points_from_line(slope, x, intercept):
    y = slope*x + intercept
    return (int(math.floor(x)), int(math.floor(y)))

def average_slope_intercept(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y)[0]
    return m, b

def get_weighted_mean(points):

    # TO DO: add weight here in terms of length
    # length = math.sqrt((x1-x2)**2 + (y1-y2)**2)

    return np.mean(points)

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1.0, phi=0.0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, phi)

def lane_finder(image):

    # bgr to hsv
    hsv = bgr_to_hsv(image)

    # filitered
    filtered = yellow_white_filter(hsv)

    # gaussian blur
    kernel_size = 7
    blur_gray = gaussian_blur(filtered, kernel_size)

    # canny
    low_threshold = 1
    high_threshold = 5
    edges = canny(blur_gray, low_threshold, high_threshold)

    # region of interest
    vertices = np.array([[(50, 550), (910,550), (510,330),(450, 330)]], dtype=np.int32)
    edges = region_of_interest(edges, vertices)

    # hough line
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/90
    threshold = 15
    min_line_len = 50
    max_line_gap = 30

    # Run Hough on edge detected image
    line_img = hough_lines(edges, rho, theta, threshold, min_line_len, max_line_gap)

    # weighted image
    weighted = weighted_img(line_img, image)
    return weighted

def yellow_white_filter(img):
    # define range of white color in HSV
    lower_white = np.array([0,0,0], dtype=np.uint8)
    upper_white = np.array([200,200,200], dtype=np.uint8)

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(img, lower_white, upper_white)

    return mask

def test_lane(img):

    hsv = bgr_to_hsv(img)
    filtered = yellow_white_filter(hsv)

    # gaussian blur
    kernel_size = 3
    blur_gray = gaussian_blur(filtered, kernel_size)

    # canny
    low_threshold = 1
    high_threshold = 5
    edges = canny(blur_gray, low_threshold, high_threshold)

    # region of interest
    vertices = np.array([[(50, 550), (910,550), (500,305),(460, 305)]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)

    # hough line
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/45
    threshold = 3
    min_line_len = 20
    max_line_gap = 20

    # Run Hough on edge detected image
    line_img = hough_lines(roi, rho, theta, threshold, min_line_len, max_line_gap)
    weighted = weighted_img(image, line_img)

    return edges

# image = mpimg.imread('segments/solidWhiteRight249.jpg')
# image = mpimg.imread('test_images/solidWhiteRight.jpg')
# image = mpimg.imread('test_images/solidYellowCurve.jpg')
# image = mpimg.imread('test_images/solidYellowCurve2.jpg')
# image = mpimg.imread('test_images/solidYellowLeft.jpg')
# image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
# plt.figure()
# plt.imshow(test_lane(image))
# plt.show()

# white_output = 'yellow-v5.mp4'
# clip1 = VideoFileClip("solidYellowLeft.mp4")
white_output = 'white-v5.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
# white_output = 'extra-v1.mp4'
# clip1 = VideoFileClip("challenge.mp4")
white_clip = clip1.fl_image(lane_finder) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
