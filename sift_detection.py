# imports
import cv2 as cv
import random
import numpy as np
import math
import time
import sys
from matplotlib import pyplot as plt


def find_matches_by_ratio(distances):
    results = []
    for x in range(distances.shape[0]):
        best, second_best = sys.maxsize, sys.maxsize
        best_index, second_best_index = (-1, -1), (-1, -1)
        for y in range(distances.shape[1]):
            if distances[x, y] <= best:
                second_best = best
                second_best_index = best_index
                best = distances[x, y]
                best_index = (x, y)
            elif distances[x, y] <= second_best:
                second_best = distances[x, y]
                second_best_index = (x, y)
        results.append((best_index, second_best_index))
    return results


def check_matches_ratio(matches, distances):
    result = []
    for match in matches:
        if distances[match[0][0], match[0][1]] < 0.8 * distances[match[1][0], match[1][1]]:
            result.append(match[0])
    return result


def check_matches_bidirectional(vec1, vec2):
    results = []
    for x in range(vec1.shape[0]):
        best = sys.maxsize
        best_index = (-1, -1)
        for y in range(vec2.shape[0]):
            distance = np.linalg.norm(vec1[x] - vec2[y])
            if distance <= best:
                best = distance
                best_index = (x, y)
        results.append(best_index)
    return results


def find_final_bidirectional_matches(bidirectional_matches1, bidirectional_matches2, kp1, kp2):
    first_matches = []
    second_matches = []
    result = []
    # take points of first match
    for match in bidirectional_matches1:
        p1 = kp1[match[0]].pt
        p2 = kp2[match[1]].pt
        first_matches.append((p1, p2, match))
    # take points of second match
    for match in bidirectional_matches2:
        p1 = kp2[match[0]].pt
        p2 = kp1[match[1]].pt
        second_matches.append((p1, p2))
    # to take matches that they are in both lists
    for p1, p2, match in first_matches:
        if (p2, p1) in second_matches:
            result.append(match)
    return result


def draw_matching_points(kp1, kp2, image1, image2, matches):
    """
    :param kp2: real points for image 2 feature pixles (vector)
    :param kp1: real points for image 1 feature pixles (vector)
    :param image1: source image
    :param image2: target image
    :param matches: a tuple of all matched description vector need to draw line between pixels
    :return: concatenate image(image1+image2) with lines between points
    """
    #random_samples_points = random.sample(matches, 75)
    random_samples_points = matches
    concatenate_image = np.concatenate((image1, image2), axis=1)
    image2_start_index = image1.shape[1]
    for point in random_samples_points:
        random_number1 = random.randint(0, 255)
        random_number2 = random.randint(0, 255)
        random_number3 = random.randint(0, 255)
        color = (random_number1, random_number2, random_number3)
        p1 = kp1[point[0]].pt
        p2 = kp2[point[1]].pt
        start_line_point = (int(p1[0]), int(p1[1]))
        finish_line_point = (int(image2_start_index+p2[0]), int(p2[1]))
        concatenate_image = cv.line(concatenate_image, start_line_point, finish_line_point, color, 2)
    return concatenate_image


def draw_key_points(image, kp):
    """

    :param image:
    :param kp:
    :return:
    """
    for KeyPoint in kp:
        random_number1 = random.randint(0, 255)
        random_number2 = random.randint(0, 255)
        random_number3 = random.randint(0, 255)
        color = (random_number1, random_number2, random_number3)
        start_point = KeyPoint.pt
        start_point = (int(start_point[0]), int(start_point[1]))
        size = KeyPoint.size
        angle = KeyPoint.angle
        end_point = (int(start_point[0] + math.cos(math.radians(angle))*size), int(start_point[1] + math.sin(math.radians(angle))*size))
        image = cv.arrowedLine(image, start_point, end_point, color, 1)
        radius = int(math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2))
        image = cv.circle(image, start_point, radius, color, 1)


def main_function():
    print("Program starting ...    Q2")
    imgUoH = cv.imread('Q2/UoH.jpg')
    pair_imgA = cv.imread('Q2/pair1_imageA.jpg')
    pair_imgB = cv.imread('Q2/pair1_imageB.jpg')

    '-------------finds key points by SIFT algorithm-------------'
    sift_detector = cv.SIFT_create()
    key_pts = sift_detector.detect(imgUoH, None)
    kp1, descriptor_vec1 = sift_detector.detectAndCompute(pair_imgA, None)
    kp2, descriptor_vec2 = sift_detector.detectAndCompute(pair_imgB, None)

    distance_between_descriptors = np.zeros((descriptor_vec1.shape[0], descriptor_vec2.shape[0]))
    for i in range(descriptor_vec1.shape[0]):
        for j in range(descriptor_vec2.shape[0]):
            distance_between_descriptors[i, j] = np.linalg.norm(descriptor_vec1[i] - descriptor_vec2[j])

    # find ratio matches
    matches = find_matches_by_ratio(distance_between_descriptors)
    higher_than_ratio_matches = check_matches_ratio(matches, distance_between_descriptors)


    # find bidirectional matches
    bidirectional_matches1 = check_matches_bidirectional(descriptor_vec1, descriptor_vec2)
    bidirectional_matches2 = check_matches_bidirectional(descriptor_vec2, descriptor_vec1)
    final_matches_bidirectional = find_final_bidirectional_matches(bidirectional_matches1, bidirectional_matches2, kp1, kp2)

    print(len(final_matches_bidirectional))
    print(len(higher_than_ratio_matches))
    # draw both matching and keypoints
    '-------------draw Key points-------------'
    draw_key_points(imgUoH, key_pts)
    cv.imshow("image UoH with keypoints", imgUoH)
    draw_key_points(pair_imgA, kp1)
    cv.imshow("image imageA with key points", pair_imgA)
    draw_key_points(pair_imgB, kp2)
    cv.imshow("image imageB with key points", pair_imgB)
    cv.waitKey(0)
    cv.destroyAllWindows()
    '-------------draw matching points-------------'
    final_image_1 = draw_matching_points(kp1, kp2, pair_imgA, pair_imgB, higher_than_ratio_matches)
    plt.imshow(final_image_1)
    plt.title('Pair1 Image In ratio test'), plt.xticks([]), plt.yticks([])
    plt.show()
    final_image_2 = draw_matching_points(kp1, kp2, pair_imgA, pair_imgB, final_matches_bidirectional)
    plt.imshow(final_image_2)
    plt.title('Pair1 Image In bidirectional matches'), plt.xticks([]), plt.yticks([])
    plt.show()


main_function()