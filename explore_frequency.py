import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage
from scipy import signal
from skimage import io
from skimage.filters import gaussian
from skimage.transform import rescale
import cv2 as cv

# pencil = io.imread('pencil.jpg', as_gray=True)
# pencil = skimage.img_as_float32(pencil)
#
#
# io.imsave('gray_pencil.jpg', skimage.img_as_ubyte(pencil))
#
filt1 = np.ones((5, 5))*1/25 # box filter
#
# filt15 = np.zeros((5,5))
# filt15[2][2] = 1
#
# filt2 = np.array([[-1/9, -1/9, -1/9],
#                   [-1/9, 2 - 1/9, -1/9],
#                   [-1/9, -1/9, -1/9]])
#
# filt3 = np.zeros((31,31))
# filt3[15][15] = 1
# filt3 = gaussian(filt3, sigma=10)
#
# filt4 = np.array([[1, 2, 1],
#                   [0, 0, 0],
#                   [-1, -2, -1]])
#
# filt5 = np.array([[1, 0, -1],
#                   [2, 0, -2],
#                   [1, 0, -1]])

# mystery = scipy.signal.signaltools.correlate2d(pencil, filt4)
# mystery = (mystery - np.min(mystery))/(np.max(mystery) - np.min(mystery))
#
# mystery2 = scipy.signal.signaltools.correlate2d(pencil, filt5)
# mystery2 = (mystery2 - np.min(mystery2))/(np.max(mystery2) - np.min(mystery2))
#
# mystery3 = np.sqrt(mystery**2 + mystery2**2)
#
# io.imsave("f_pencil.jpg", skimage.img_as_ubyte(mystery3))

# edge_sobel = skimage.filters.sobel(pencil)
# io.imsave("sobel_plant.jpg", skimage.img_as_ubyte(edge_sobel))
#
#
# sp_pencil = skimage.util.random_noise(pencil, "s&p")
# # io.imsave("sp_pencil.jpg", skimage.img_as_ubyte(sp_pencil))
#
# sp_blurred = scipy.signal.signaltools.correlate2d(sp_pencil, filt1)
# io.imsave("sp_blurred.jpg", skimage.img_as_ubyte(sp_blurred))
#
# med_pencil = skimage.filters.median(sp_pencil)
# io.imsave("median_sppencil.jpg", skimage.img_as_ubyte(med_pencil))
#
# canny_pencil = skimage.feature.canny()



image = cv.imread("pencil.jpg", cv.IMREAD_GRAYSCALE)
image2 = cv.imread("sp_blurred.jpg", cv.IMREAD_GRAYSCALE)

kaze = cv.AKAZE_create()
kps, descs = kaze.detectAndCompute(image, None)
kps2, descs2 = kaze.detectAndCompute(image2, None)

# def draw_kps(im, kps, col, th):
#     for kp in kps:
#         x = np.int(kp.pt[0])
#         y = np.int(kp.pt[1])
#         size = np.int(kp.size)
#         cv.circle(im, (x, y), size, col, thickness=th, lineType=8, shift=0)
#     plt.imshow(im, cmap="gray")
#     return im
#
# im_circles = draw_kps(image.copy(), kps, (0, 255, 0), 2)
# plt.show()

bf = cv.BFMatcher()
matches = bf.knnMatch(descs, descs2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

img3 = cv.drawMatchesKnn(image, kps, image2, kps2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite('matches.jpg', img3)