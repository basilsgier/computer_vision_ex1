import cv2 as cv
import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np
from math import *
import math

max_lowThreshold = 100
ratio = 4
kernel_size = 3


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


def show_hough_line(img, accumulator, thetas, rhos, save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(
        accumulator, cmap='jet',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')


def CannyThreshold(val, src, src_gray):
    low_threshold = val
    img_blur = cv.blur(src_gray, (3, 3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:, :, None].astype(src.dtype))
    return dst


def rotate(origin, point, angle):
    CosineAngle = math.cos(angle)
    SinAngle = math.sin(angle)
    ox, oy = origin
    px, py = point
    qx = ox + CosineAngle * (px - ox) - SinAngle * (py - oy)
    qy = oy + SinAngle * (px - ox) + CosineAngle * (py - oy)
    return [int(qx), int(qy)]


def gradiantX(edge_img):
    return np.array(sig.convolve2d(edge_img, [[-1, 1]], mode='same'), dtype='float')


def gradiantY(edge_img):
    return np.array(sig.convolve2d(edge_img, [[-1], [1]], mode='same'), dtype='float')


def init_hough_space(edge_img, L, originalImage, gradientImage):
    hough_space = np.zeros((edge_img.shape[1], edge_img.shape[0], 120), dtype='uint32')
    print(hough_space.shape)
    for x in range(edge_img.shape[0]):
        for y in range(edge_img.shape[1]):
            if edge_img[x, y] != 0:
                for Edge in range(int(-L / 2), int(L / 2)):
                    angle = gradientImage[x, y]
                    Snormal = math.sin(angle)
                    Cnormal = math.cos(angle)
                    CedgeNormal = math.cos(math.radians(90) + angle)
                    SedgeNormal = math.sin(math.radians(90) + angle)
                    a = int(y + Edge * CedgeNormal + (math.sqrt(3) / 6) * L * Cnormal)
                    b = int(x + Edge * SedgeNormal + (math.sqrt(3) / 6) * L * Snormal)
                    if (a >= 0) and (a < originalImage.shape[0]) and (b >= 0) and (b < originalImage.shape[1]):
                        hough_space[b, a, int(math.degrees(angle) % 120)] += 1
    return hough_space


def Detect_triangles_and_draw(hough_space, image, color, threshold, len_of_edge):
    for b in range(hough_space.shape[0]):
        for a in range(hough_space.shape[1]):
            for theta in range(hough_space.shape[2]):
                if hough_space[b, a, theta] >= threshold:
                    VertexA = [a + ((sqrt(3) / 3) * len_of_edge), b]
                    VertexB = [a - ((sqrt(3) / 6) * len_of_edge), b - (len_of_edge / 2)]
                    VertexC = [a - ((sqrt(3) / 6) * len_of_edge), b + (len_of_edge / 2)]
                    VertexA = rotate([a, b], VertexA, math.radians(theta))
                    VertexB = rotate([a, b], VertexB, math.radians(theta))
                    VertexC = rotate([a, b], VertexC, math.radians(theta))
                    pts = np.array([VertexA, VertexB, VertexC])
                    pts = pts.reshape((-1, 1, 2))
                    isClosed = True
                    image = cv.polylines(image, [pts], isClosed, color, 2)
    return image


def process_img_q1(image, len_of_edge, vote_threshold, canny_threshold):
    print("Starting the program .. Q1 ")

    # Part 1 uses CV Canny to find Edges map and use rgb2gray to make it grayscale.
    image_gray = rgb2gray(image)
    image_edges = CannyThreshold(canny_threshold, image, image_gray)
    image_edges = rgb2gray(image_edges)
    plt.imshow(image_edges)
    plt.show()

    # filter to change the edge image to binary 0 or 255 white or black
    # for a in range(image_edges.shape[0]):
    #     for b in range(image_edges.shape[1]):
    #         if image_edges[a][b] != 0:
    #             image_edges[a][b] = 255

    # part 2 Find gradiant for Edges map
    image_grad_x = gradiantX(image_gray)
    image_grad_y = gradiantY(image_gray)
    image_grad = np.arctan2(image_grad_y, image_grad_x)

    # part 3 and 4 init the hough space and vote for evrey triangle
    hough_space = init_hough_space(image_edges, len_of_edge, image_gray, image_grad)

    # part 5 draw all detected triangle that they have vote more than vote_threshould
    return Detect_triangles_and_draw(hough_space, image, (0, 255, 0), vote_threshold, len_of_edge)


# # first image image name image003
# len_of_edge = 100
# vote_threshold = 5
# canny_threshold = 100
# image003 = cv.imread("Q1/triangles_1/image003.jpg")
# image003_after_detection = process_img_q1(image003, len_of_edge, vote_threshold, canny_threshold)
#
# plt.subplot(121), plt.imshow(image003, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(image003_after_detection, cmap='gray')
# plt.title('After detection'), plt.xticks([]), plt.yticks([])
# plt.show()
# cv.waitKey(0)


# # second image image name image002
# len_of_edge = 12
# vote_threshold = 5
# canny_threshold = 25
#
# image002 = cv.imread("Q1/triangles_1/image002.jpg")
# for x in range(image002.shape[0]):
#     for y in range(image002.shape[1]):
#         if image002[x][y][0] < 180:
#             image002[x][y] = (0, 0, 0)
#         else:
#             image002[x][y] = (255, 255, 255)
#
#
# plt.imshow(image002)
# plt.show()
# image002_after_detection = process_img_q1(image002, len_of_edge, vote_threshold, canny_threshold)
#
# plt.subplot(121), plt.imshow(image002, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(image002_after_detection, cmap='gray')
# plt.title('After detection'), plt.xticks([]), plt.yticks([])
# plt.show()
# cv.waitKey(0)


# first image in triangles_2 name image012
len_of_edge = 110
vote_threshold = 10
canny_threshold = 100
image003 = cv.imread("Q1/triangles_2/image012.jpg")
image003_after_detection = process_img_q1(image003, len_of_edge, vote_threshold, canny_threshold)

plt.subplot(121), plt.imshow(image003, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(image003_after_detection, cmap='gray')
plt.title('After detection'), plt.xticks([]), plt.yticks([])
plt.show()
cv.waitKey(0)

# # second image in triangles_2 name image006
# len_of_edge = 115
# vote_threshold = 11
# canny_threshold = 120
# image003 = cv.imread("Q1/triangles_2/image006.jpg")
# plt.imshow(image003)
# plt.show()
# image003_after_detection = process_img_q1(image003, len_of_edge, vote_threshold, canny_threshold)
#
# plt.subplot(121), plt.imshow(image003, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(image003_after_detection, cmap='gray')
# plt.title('After detection'), plt.xticks([]), plt.yticks([])
# plt.show()
# cv.waitKey(0)

