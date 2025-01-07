import numpy as np
from typing import List, Tuple
import cv2

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

'''
Please do Not change or add any imports. 
'''

#task1

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from xyz to XYZ.

    '''
    #rot_xyz2XYZ = np.eye(3).astype(float)

    # Your implementation
    
    rot_xyz2XYZ = np.matmul(rotate_matrix_along_z(gamma), np.matmul(rotate_matrix_along_x(beta), rotate_matrix_along_z(alpha)))

    return rot_xyz2XYZ


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from XYZ to xyz.

    '''
    rot_XYZ2xyz = np.eye(3).astype(float)

    # Your implementation
    
    rot_XYZ2xyz = np.matmul(rotate_matrix_along_z(-alpha), np.matmul(rotate_matrix_along_x(-beta), rotate_matrix_along_z(-gamma)))
    
    return rot_XYZ2xyz

"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()" functions are the only 2 function that will be called in task1.py.
"""

# Your functions for task1

def rotate_matrix_along_x(angle):
    radians = np.radians(angle)
    
    rot_x = np.array([[1,0,0],[0, np.cos(radians), -np.sin(radians)],[0, np.sin(radians), np.cos(radians)]])
    
    return rot_x

def rotate_matrix_along_z(angle):
    radians = np.radians(angle)
    
    rot_z = np.array([[np.cos(radians), -np.sin(radians), 0],[np.sin(radians), np.cos(radians),0], [0,0,1]])
    
    return rot_z




#--------------------------------------------------------------------------------------------------------------
# task2:

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Args: 
        image: Input image of size MxNx3. M is the height of the image. N is the width of the image. 3 is the channel of the image.
    Return:
        A numpy array of size 32x2 that represents the 32 checkerboard corners' pixel coordinates. 
        The pixel coordinate is defined such that the of top-left corner is (0, 0) and the bottom-right corner of the image is (N, M). 
    '''
    #img_coord = np.zeros([32, 2], dtype=float)

    # Your implementation
    PATTERN = (8,4)
    gray_img = cvtColor(image, COLOR_BGR2GRAY)
    ret, corners = findChessboardCorners(gray_img, PATTERN, None)
    
    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)
    if ret:
        img_coord = cornerSubPix(gray_img, corners, (11,11), (-1,-1), criteria)
        img_coord = img_coord.reshape(32,2)

    return img_coord

def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    '''
    You can output the world coord manually or through some algorithms you design. Your output should be the same order with img_coord.
    Args: 
        img_coord: The image coordinate of the corners. Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.
    Return:
        A numpy array of size 32x3 that represents the 32 checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image. The output results should be in milimeters.
    '''
    #world_coord = np.zeros([32, 3], dtype=float)

    # Your implementation
    world_coord = []
    for z_coord in [40,30, 20, 10]:
        for coord in [-30,-20,-10, 0 ,10, 20, 30, 40]:
            if coord < 0:
                coord_point  = [0 ,-coord, z_coord]
            elif coord >=0:
                coord_point = [coord, 0, z_coord]
            world_coord.append(coord_point)
    world_coord = np.array(world_coord)

    return world_coord


def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 32 point to calculate the intrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''

    #fx: float = 0
    #fy: float = 0
    #cx: float = 0
    #cy: float = 0

    # Your implementation
    P = solving_for_intrinsic_and_extrinsic_matrix(img_coord, world_coord)
    
    #fx = K[0,0]
    #fy = K[1,1]
    
    #cx = K[0,2]
    #cy = K[1,2]
    
    m1 = P[0, :3]
    m2 = P[1, :3]
    m3 = P[2,:3]
    
    
    cx = m1.T @ m3
    cy = m2.T @ m3
    
    fx = np.sqrt(m1.T@m1 - np.square(cx))
    fy = np.sqrt(m2.T@m2 - np.square(cy))
    
    return fx, fy, cx, cy


def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Use the image coordinates, world coordinates of the 32 point and the intrinsic parameters to calculate the extrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        R: The rotation matrix of the extrinsic parameters. It is a 3x3 numpy array.
        T: The translation matrix of the extrinsic parameters. It is a 1-dimensional numpy array with length of 3.
    '''

    #R = np.eye(3).astype(float)
    #T = np.zeros(3, dtype=float)

    # Your implementation
    P = solving_for_intrinsic_and_extrinsic_matrix(img_coord, world_coord)
    
    fx, fy, ox, oy = find_intrinsic(img_coord, world_coord)
    
    r31, r32, r33, tz = P[2]
    
    r11 = (P[0,0] - ox*r31)/fx
    r12 = (P[0,1] - ox*r32)/fx
    r13 = (P[0,2] - ox*r33)/fx
    tx = (P[0,3] - ox*tz)/fx
    
    r21 = (P[1,0] - oy*r31)/fy
    r22 = (P[1,1] - oy*r32)/fy
    r23 = (P[1,2] - oy*r33)/fy
    ty = (P[1,3] - oy*tz)/fy
    
    
    R = np.array([[r11, r12, r13],[r21, r22, r23], [r31, r32, r33]])
    
    t = np.array([tx, ty, tz])
    

    return R, t


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above 4 functions are the only ones that will be called in task2.py.
"""

# Your functions for task2

def solving_for_intrinsic_and_extrinsic_matrix(img_coord, world_coord):
    A = []
    for idx in range(len(img_coord)):
        x, y = img_coord[idx]
        X,Y,Z = world_coord[idx]
    
        row1 = [X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x]
        row2 = [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y]
        A.append(row1)
        A.append(row2)
    
    A = np.array(A)
    
    """
    A = U*S*V.T (SVD)
    A.T*A = (U*S*V.T).T * (U*S*V.T)
    A.T*A = (V.T).T * S.T * U.T * U * S * V.T
    A.T*A = V * S.T * (U.T * U) * S * V.T
    
    as U is unitary matrix U.T*U = I
    
    A.T*A = V * S.T * (I) * S * V.T
    A.T*A = V*(S.T*S)*V.T
    
    As S is a diagnol matrix S.T*S = S^2
    
    A.T*A = V*S^2*V.T
    
    As A.T*A is a square matrix we can do Eigen decomposition.
    
    Q*L*Q.T = V*S^2*V.T
    
    Q = V
    L = S^2
    
    V columns contains Eigen Vectors and V.T rows contain Eigen Vectors.
    As Singular values or in order the last singular values is the minimum and last eigen vector is
    its corresponding vector
    """
    _ , _ ,  Vt = np.linalg.svd(A)
    
    P = Vt[-1].reshape(3,4)
    
    scale_factor = 1/np.sqrt(np.square(P[2,0])+np.square(P[2,1])+np.square(P[2,2]))
    
    
    return scale_factor*P

#---------------------------------------------------------------------------------------------------------------------