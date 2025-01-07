# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 3. Not following the project guidelines will result in a 10% reduction in grades
# 4 . If you want to show an image for debugging, please use show_image() function in helper.py.
# 5. Please do NOT save any intermediate files in your final submission.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import array as arr


def parse_args():
    parser = argparse.ArgumentParser(description="cse 573 homework 4.")
    parser.add_argument(
        "--input_path", type=str, default="data/images_panaroma",
        help="path to images for panaroma construction")
    parser.add_argument(
        "--output_overlap", type=str, default="./task2_overlap.txt",
        help="path to the overlap result")
    parser.add_argument(
        "--output_panaroma", type=str, default="./task2_result.png",
        help="path to final panaroma image ")

    args = parser.parse_args()
    return args

def extract_features(image):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image)
    # Detect keypoints and compute descriptors
    key_pointers , descriptors = sift.detectAndCompute(image, None)
    return key_pointers, descriptors

def match_features(descriptors1, descriptors2, min_distance_threshold):
    
    d1 = descriptors1[:, np.newaxis, :]
    d2 = descriptors2[np.newaxis, :, :]

    
    distances = np.linalg.norm(d1 - d2, axis=2)

    min_distances = np.min(distances, axis=1)
    closest_indices = np.argmin(distances, axis=1)

    valid_matches = min_distances <= min_distance_threshold

    matches = [[i,closest_indices[i]] for i in range(len(descriptors1)) if valid_matches[i]]

    return matches


def find_image_pairs(overlap_matrix):
    pairs = []
    n = len(overlap_matrix)

    for i in range(n):
        for j in range(i + 1, n):
            if overlap_matrix[i][j] == 1:
                # Add the pair (i, j) if the images overlap
                pairs.append((i, j))

    return pairs


def get_translation_matrix(img, H):
    
    # Define corners in homogeneous coordinates
    height, width = img.shape[:2]
    corners = np.array([
        [0, 0, 1],                               # Top-left
        [0, height - 1, 1],                       # Bottom-left
        [width - 1, 0, 1],                        # Top-right
        [width - 1, height - 1, 1]                # Bottom-right
    ])

    # Transform corners using the homography matrix
    transformed_corners = np.matmul(H,corners.T).T

    # Normalize to convert from homogeneous coordinates
    transformed_corners = transformed_corners / transformed_corners[:, 2][:, np.newaxis]

    # Calculate translation values
    translation_x = int(max(-transformed_corners[:, 0].min(), 0))
    translation_y = int(max(-transformed_corners[:, 1].min(), 0))

    # Calculate dimensions of the transformed image
    new_H = int(max(transformed_corners[:, 1].max(), 0)) + translation_y
    new_W = int(max(transformed_corners[:, 0].max(), 0)) + translation_x

    # Create translation matrix
    M = np.array([[1, 0, translation_x], [0, 1, translation_y], [0, 0, 1]], dtype=float)

    return M, new_H, new_W


def stitch_images(img1, img2, key_pointers1, key_pointers2, matches):
    
    img1_pts = []
    img2_pts = []
    
    for m in matches:
        img1_pts.append(key_pointers1[m[0]].pt)
        img2_pts.append(key_pointers2[m[1]].pt)
        
    img1_pts = np.array(img1_pts)
    img2_pts = np.array(img2_pts)
    
    # Compute the homography matrix
    H, _ = cv2.findHomography(img2_pts, img1_pts, cv2.RANSAC, 5.0)
    
    
    translation_matrix, new_H, new_W = get_translation_matrix(img1, H)
    
    H_img2 = np.matmul(translation_matrix, H)
    H_img1 = np.matmul(translation_matrix,np.identity(3))
    
    warpim2 = cv2.warpPerspective(img2, H_img2, (new_W, new_H))
    warpim1 = cv2.warpPerspective(img1, H_img1, (new_W, new_H))
  
    blended_img = np.where(warpim1>0, warpim1, warpim2)
    return blended_img


def stitch(inp_path, imgmark, N=4, savepath=''): 
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'{inp_path}/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    overlap_arr = np.eye(len(imgs), dtype=int)
    descriptors = {}
    key_pointers = {}
    matched_points_set = {}
    stitched_image = None
    processed_set = set()
    for i, img1 in enumerate(imgs):
        for j, img2 in enumerate(imgs):
            if i != j:
                # Extract features and match
                if i not in descriptors.keys():
                    keys, descs = extract_features(img1)
                    key_pointers[i] = keys
                    descriptors[i] = descs
                if j not in descriptors.keys():
                    keys,descs= extract_features(img2)
                    key_pointers[j] = keys
                    descriptors[j] = descs
                    
                matches = match_features(descriptors[i], descriptors[j], 50)
                
                matched_points_set[(i,j)] = matches
                
                min_match_threshold = 10 # Value to tune
                if len(matches) > min_match_threshold:
                    overlap_arr[i, j] = 1
                    if stitched_image is None:
                        stitched_image = stitch_images(imgs[i], imgs[j], key_pointers[i], key_pointers[j], matched_points_set[(i,j)])
                        processed_set.add(i)
                        processed_set.add(j)
                    elif stitched_image is not None and j not in processed_set:
                        key_pointers_stitiched_image, descriptors_stitiched_image = extract_features(stitched_image)
                        matched_points = match_features(descriptors_stitiched_image, descriptors[j], 50)
                        stitched_image = stitch_images(stitched_image, imgs[j], key_pointers_stitiched_image, key_pointers[j], matched_points)
                        processed_set.add(j)
                else:
                    overlap_arr[i, j] = 0
                
    with open('overlap_arr.json', 'w') as f:
        json.dump(overlap_arr.tolist(), f)
    
    if stitched_image is not None:
        cv2.imwrite(savepath, stitched_image)
                
    return overlap_arr
    
if __name__ == "__main__":
    #task2
    args = parse_args()
    overlap_arr = stitch(args.input_path, 't2', N=4, savepath=f'{args.output_panaroma}')
    with open(f'{args.output_overlap}', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    
