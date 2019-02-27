'''  
Author: Dewang Sultania, Nanda Kishore 
  Date created: 10/31/2018
'''

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from scipy import signal

from corner_detector import corner_detector
from anms import anms
from feat_desc import feat_desc
from feat_match import feat_match
from est_homography import est_homography
from ransac_est_homography import ransac_est_homography
from seamlessCloningPoisson import seamlessCloningPoisson

ALPHA_BLENDING = 0
GRADIENT_BLENDING = 1

def rgb2gray(I_rgb):
    r, g, b = I_rgb[:, :, 0], I_rgb[:, :, 1], I_rgb[:, :, 2]
    I_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return I_gray

class Helper:

    def __init__(self, img_left, img_center, img_right, stitching_mode):
        self.img_left = img_left
        self.img_center = img_center
        self.img_right = img_right
        self.img_left_gray = rgb2gray(img_left)
        self.img_center_gray = rgb2gray(img_center)
        self.img_right_gray = rgb2gray(img_right)
        self.stitching_mode = stitching_mode

    def corner_detectors(self):
        self.cimg_left = corner_detector(self.img_left_gray)
        self.cimg_center = corner_detector(self.img_center_gray)
        self.cimg_right = corner_detector(self.img_right_gray)


    def anms_outputs(self):
        self.x_left, self.y_left, self.rmax_left = anms(\
        self.cimg_left, 250)
        self.x_center, self.y_center, self.rmax_center = anms(\
        self.cimg_center, 250)
        self.x_right, self.y_right, self.rmax_right = anms(\
        self.cimg_right, 250)

    def feature_outputs(self):
        self.features_left = feat_desc(self.img_left_gray, \
        self.x_left, self.y_left)
        self.features_center = feat_desc(self.img_center_gray, \
        self.x_center, self.y_center)
        self.features_right = feat_desc(self.img_right_gray, \
        self.x_right, self.y_right)

    def matches(self):
        self.matches_left_center = feat_match(self.features_left,\
        self.features_center)
        self.matches_center_right = feat_match(self.features_right,\
        self.features_center)

    def plots(self):

        plt.title('ANMS output for Image 1')
        plt.imshow(self.img_left)
        plt.scatter(self.x_left, self.y_left, s=8)
        plt.show()
        plt.title('ANMS output for Image 2')
        plt.imshow(self.img_center)
        plt.scatter(self.x_center, self.y_center, s=8)
        plt.show()
        plt.title('ANMS output for Image 3')
        plt.imshow(self.img_right)
        plt.scatter(self.x_right, self.y_right, s=8)
        plt.show()
        plt.title('Feature matching before RANSAC for Image 1 and Image 2')
        plt.scatter(self.x_left, self.y_left, s=8)
        plt.scatter(self.x_center+self.img_left.shape[1], self.y_center, s=8)
        plt.imshow(np.concatenate((self.img_left, self.img_center), axis=1))
        for i in range(self.features_left.shape[1]):
            if self.matches_left_center[i]!=-1:
                plt.plot([self.x_left[i], \
                self.x_center[self.matches_left_center[i]]+self.img_left.shape[1]], \
                [self.y_left[i], self.y_center[self.matches_left_center[i]]], 'r')
        plt.show()
        plt.title('Feature matching after RANSAC for Image 1 and Image 2')
        plt.imshow(np.concatenate((self.img_left, self.img_center), axis=1))
        plt.scatter(self.inlier_x2+self.img_left.shape[1], self.inlier_y2, s=8)
        plt.scatter(self.inlier_x1, self.inlier_y1, s=8)
        for i in range(self.inlier_x1.shape[0]):
            plt.plot([self.inlier_x1[i], self.inlier_x2[i]+self.img_left.shape[1]], \
            [self.inlier_y1[i], self.inlier_y2[i]], 'r')
        plt.show()
        plt.title('Feature matching before RANSAC for Image 2 and Image 3')
        plt.imshow(np.concatenate((self.img_center, self.img_right), axis=1))
        plt.scatter(self.x_center, self.y_center, s=8)
        plt.scatter(self.x_right+self.img_center.shape[1], self.y_right, s=8)
        for i in range(self.features_center.shape[1]):
             if self.matches_center_right[i] != -1:
                 plt.plot([self.x_right[i]+self.img_center.shape[1], \
                 self.x_center[self.matches_center_right[i]]], [self.y_right[i],\
                 self.y_center[self.matches_center_right[i]]], 'r')
        plt.show()
        plt.title('Feature matching after RANSAC for Image 2 and Image 3')
        plt.imshow(np.concatenate((self.img_center, self.img_right), axis=1))
        plt.scatter(self.inlier_x2_center, self.inlier_y2_center, s=8)
        plt.scatter(self.inlier_x1_right+self.img_center.shape[1], self.inlier_y1_right, s=8)
        for i in range(self.inlier_x1_right.shape[0]):
            plt.plot([self.inlier_x2_center[i], self.inlier_x1_right[i]+\
            self.img_center.shape[1]], [self.inlier_y2_center[i], self.inlier_y1_right[i]], 'r')
        plt.show()

    def stitch_left(self):


        match = self.matches_left_center
        self.x1 = self.x_left[match!=-1]
        self.y1 = self.y_left[match!=-1]
        self.x2 = self.x_center[match[match!=-1]].reshape(-1)
        self.y2 = self.y_center[match[match!=-1]].reshape(-1)
        H, inlier_inds = ransac_est_homography(self.x1, self.y1, self.x2, self.y2\
        , 0.5)

        self.inlier_x1 = self.x1[inlier_inds].reshape(-1)
        self.inlier_x2 = self.x2[inlier_inds].reshape(-1)
        self.inlier_y1 = self.y1[inlier_inds].reshape(-1)
        self.inlier_y2 = self.y2[inlier_inds].reshape(-1)
        H = est_homography(self.inlier_x1, self.inlier_y1, self.inlier_x2, \
        self.inlier_y2)


        corner_pts = np.dot(H, np.array([[0, \
        self.img_left.shape[1], self.img_left.shape[1], 0], \
        [0, 0, self.img_left.shape[0], self.img_left.shape[0]], [1, 1, 1, 1]]))

        corner_pts = corner_pts[:2,:]/corner_pts[2,:]

        top_padding = np.int32(np.amin(corner_pts[1,:]))
        bottom_padding = np.int32(np.amax(corner_pts[1,:]))
        left_padding = np.int32(np.amin(corner_pts[0,:]))
        right_padding = np.int32(np.amax(corner_pts[0,:]))
        paddings = [top_padding, bottom_padding, left_padding, right_padding]
        img_left_interpolator = RegularGridInterpolator((np.arange(\
        self.img_left.shape[0]), np.arange(self.img_left.shape[1])), \
        self.img_left, bounds_error=False, fill_value=0)

        output_shape = (bottom_padding - top_padding, \
        right_padding - left_padding)
        H_inv = np.linalg.inv(H)

        x,y = np.meshgrid(np.arange(top_padding, bottom_padding), \
        np.arange(left_padding, right_padding))
        x = x.flatten()
        y = y.flatten()
        pstns = np.block([[y], [x], [np.ones((1, x.size))]])
        new_pstns = np.dot(H_inv, pstns)
        new_pstns = new_pstns[:2]/new_pstns[2]
        x = new_pstns[0,:].reshape(output_shape[0], output_shape[1], order='F')
        y = new_pstns[1,:].reshape(output_shape[0], output_shape[1], order='F')
        pts = np.dstack((y,x))
        img_left_warped = np.uint8(img_left_interpolator(pts))

        abs_top = min(paddings[0], 0)
        abs_bottom = max(paddings[1], self.img_center.shape[0])
        height = np.abs(abs_bottom - abs_top)
        abs_left = min(paddings[2],0)
        abs_right = max(self.img_center.shape[1], paddings[3])
        width = np.abs(abs_right - abs_left)

        img_left_padded = np.zeros((height, width, 3))
        img_left_padded[paddings[0]-abs_top:paddings[1]-abs_top, \
        paddings[2]-abs_left:paddings[3]-abs_left, :] = img_left_warped
        img_center_padded = np.zeros((height, width, 3))
        img_center_padded[0-abs_top:self.img_center.shape[0]-abs_top, \
        0-abs_left:self.img_center.shape[1]-abs_left:, :] = self.img_center
        paddings = [abs_top, abs_bottom, abs_left, abs_right]

        overlapping_indices = np.logical_and([img_left_padded != 0], \
        [img_center_padded != 0])[0]
        if self.stitching_mode == GRADIENT_BLENDING:
            mask = overlapping_indices[:,:,0]
            src_img = img_center_padded.copy()
            target_img = img_center_padded + img_left_padded
            target_img[overlapping_indices] = 0
            img_stitched = seamlessCloningPoisson(src_img, target_img, mask, 0, 0)
        else:
            img_center_padded[overlapping_indices] = img_center_padded[\
            overlapping_indices]/2
            img_left_padded[overlapping_indices] = img_left_padded[\
            overlapping_indices]/2
            img_stitched = np.uint8((img_center_padded + img_left_padded))

        return img_stitched, paddings

    def stitch_right(self):


        match = self.matches_center_right

        self.x1_right = self.x_right[match!=-1]
        self.y1_right = self.y_right[match!=-1]
        self.x2_center = self.x_center[match[match!=-1]].reshape(-1)
        self.y2_center = self.y_center[match[match!=-1]].reshape(-1)
        H, inlier_inds = ransac_est_homography(self.x1_right, self.y1_right, \
        self.x2_center, self.y2_center, 1)



        self.inlier_x1_right = self.x1_right[inlier_inds].reshape(-1)
        self.inlier_y1_right = self.y1_right[inlier_inds].reshape(-1)
        self.inlier_x2_center = self.x2_center[inlier_inds].reshape(-1)
        self.inlier_y2_center = self.y2_center[inlier_inds].reshape(-1)
        H = est_homography(self.inlier_x1_right, self.inlier_y1_right, \
        self.inlier_x2_center, self.inlier_y2_center)


        corner_pts = np.dot(H, np.array([[0, self.img_right.shape[1], \
        self.img_right.shape[1], 0], [0, 0, self.img_right.shape[0], self.\
        img_right.shape[0]], [1, 1, 1, 1]]))
        corner_pts = corner_pts[:2,:]/corner_pts[2,:]

        top_padding = np.int32(np.amin(corner_pts[1,:]))
        bottom_padding = np.int32(np.amax(corner_pts[1,:]))
        left_padding = np.int32(np.amin(corner_pts[0,:]))
        right_padding = np.int32(np.amax(corner_pts[0,:]))
        paddings = [top_padding, bottom_padding, left_padding, right_padding]

        img_right_interpolator = RegularGridInterpolator((np.arange(\
        self.img_right.shape[0]), \
        np.arange(self.img_right.shape[1])), self.img_right, bounds_error=False,\
        fill_value=0)

        output_shape = (bottom_padding - top_padding, right_padding - left_padding)
        H_inv = np.linalg.inv(H)
        x,y = np.meshgrid(np.arange(top_padding, bottom_padding), \
        np.arange(left_padding, right_padding))
        x = x.flatten()
        y = y.flatten()
        pstns = np.block([[y], [x], [np.ones((1, x.size))]])
        new_pstns = np.dot(H_inv, pstns)
        new_pstns = new_pstns[:2]/new_pstns[2]
        x = new_pstns[0,:].reshape(output_shape[0], output_shape[1], order='F')
        y = new_pstns[1,:].reshape(output_shape[0], output_shape[1], order='F')
        pts = np.dstack((y,x))
        img_right_warped = np.uint8(img_right_interpolator(pts))

        abs_top = min(0, paddings[0])
        abs_bottom = max(self.img_right.shape[0], paddings[1])
        height = np.abs(abs_bottom - abs_top)
        abs_left = min(0,paddings[2])
        abs_right = max(self.img_right.shape[1],paddings[3])
        width = np.abs(abs_right - abs_left)
        img_center_padded = np.zeros((height, width, 3))
        img_center_padded[0-abs_top:self.img_center.shape[0]-abs_top, \
        0-abs_left:self.img_center.shape[1]-abs_left, :] = self.img_center
        img_right_padded = np.zeros((height, width, 3))
        img_right_padded[paddings[0]-abs_top:paddings[1]-abs_top,paddings[2]-\
        abs_left:paddings[3]-abs_left,:] = img_right_warped
        paddings = [abs_top, abs_bottom, abs_left, abs_right]

        overlapping_indices = np.logical_and([img_center_padded != 0], \
        [img_right_padded != 0])[0]
        if self.stitching_mode == GRADIENT_BLENDING:
            mask = overlapping_indices[:,:,0]
            src_img = img_center_padded.copy()
            target_img = img_center_padded + img_right_padded
            target_img[overlapping_indices] = 0
            img_stitched = seamlessCloningPoisson(src_img, target_img, mask, 0, 0)
        else:
            img_right_padded[overlapping_indices] = img_right_padded[overlapping_indices]/2
            img_center_padded[overlapping_indices] = img_center_padded[overlapping_indices]/2
            img_stitched = np.uint8((img_right_padded + img_center_padded))

        return img_stitched, paddings

    def execute(self):
        self.corner_detectors()
        self.anms_outputs()
        self.feature_outputs()
        self.matches()
        img_left_stitched, paddings_left = self.stitch_left()
        img_right_stitched, paddings_right = self.stitch_right()
        self.plots()
        return img_left_stitched, paddings_left, img_right_stitched, paddings_right
