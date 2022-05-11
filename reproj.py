import cv2
import numpy as np

# def reproject(R, X, t):
#     P = R.dot(X) + t
#     p = -P / P[2]
    

def main():
    with open('reproj_data.txt', 'r') as data:
        lines = data.readlines()

    image_feature_points = []
    feature_positions = []
    img = cv2.imread('image_0.png')
    for i in range(0, len(lines), 2):
        feature_loc = lines[i]
        feature_pos = lines[i+1]
        
        image_x, image_y = map(int, map(float, feature_loc.split()))
        image_feature_points.append((image_x, image_y))

        img = cv2.circle(img, (image_x, image_y), radius=3, color=(0, 0, 255), thickness=-1)
        
        feature_x, feature_y, feature_z = map(float, feature_pos.split())
        feature_positions.append((feature_x, feature_y, feature_z))
        
        # R = np.array([0, 0, 0.617324])
        # t = np.array([-7.90312, -29.3841, 1])
        R = np.array([0.0, 0.0, 0.0])
        t = np.array([0.0, 0.0, 0.0])
        K = np.array([[608.228882, 0, 641.786133],
                      [0, 608.156616, 366.616272],
                      [0, 0, 1] ])
        dist_coeffs = np.array([0.411396, -2.710792, 0, 0, 1.644717, 0.291122, -2.525489, 1.563623])
    
    reprojected_points, _ = cv2.projectPoints(np.array(feature_positions), R, t, K, dist_coeffs)
    reprojected_points = reprojected_points[:, 0, :]
    print(reprojected_points.shape)

    for rp in reprojected_points:
        print(int(rp[0]), int(rp[1]))
        img = cv2.circle(img, (int(rp[0]), int(rp[1])), radius=3, color=(255, 0, 0), thickness=-1)

    cv2.imshow('img', img)
    cv2.waitKey()

main()