import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("Q3/Dylan.jpg")
frame = cv.imread("Q3/frames.jpg")


def computeAffineTransform(img, frame, srcPoints_Affine, dstPoints_Affine):
    """ cv.estimateAffine2D computes an optimal affine transformation between two 2D point sets """
    trans = cv.estimateAffine2D(srcPoints_Affine, dstPoints_Affine)[0]
    """ cv.warpAffine applies an affine transformation to an image """
    aff = cv.warpAffine(img, trans, (frame.shape[1], frame.shape[0]))
    return aff


def computeHomographyTransform(img, frame, srcPoints_Homog, dstPoints_Homog):
    """cv.getPerspectiveTransform calculates a perspective transform from four pairs of the corresponding points. """
    trans = cv.getPerspectiveTransform(srcPoints_Homog, dstPoints_Homog)
    """ cv.warpPerspective applies a perspective transformation to an image. """
    Homog = cv.warpPerspective(img, trans, (frame.shape[1], frame.shape[0]))
    return Homog


def main():
    img = cv.imread("Q3/Dylan.jpg")
    frame = cv.imread("Q3/frames.jpg")

    srcPoints_Affine = np.array([
        [0, 0],
        [640, 0],
        [640, 480]
    ])
    dstPoints_Affine = np.array([
        [549, 219],
        [845, 59],
        [902, 294]
    ])
    srcPoints_Homog = np.array([
        [0, 0],
        [640, 0],
        [640, 480],
        [0, 480]
    ]).astype(np.float32)
    dstPoints_Homog = np.array([
        [195, 57],
        [494, 159],
        [430, 502],
        [36, 182]
    ]).astype(np.float32)
    affine_warped_img = computeAffineTransform(img, frame, srcPoints_Affine, dstPoints_Affine)
    Homog_warped_img = computeHomographyTransform(img, frame, srcPoints_Homog, dstPoints_Homog)
    cv.imshow("im1", affine_warped_img + Homog_warped_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
