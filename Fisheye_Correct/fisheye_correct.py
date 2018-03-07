import cv2
import numpy as np

PI = np.pi
OUT_HEIGHT = 480

# CAM_MOD
EQUIDISTANCE = 0
USER_DEFINE = 1


def rotate3d(point3d, dim='x', rotate_angle=0):
    cosine = np.cos(rotate_angle)
    sine = np.sin(rotate_angle)
    rx = np.array([[1, 0, 0], [0, cosine, sine], [0, -sine, cosine]])
    ry = np.array([[cosine, 0, -sine], [0, 1, 0], [sine, 0, cosine]])
    rz = np.array([[cosine, sine, 0], [-sine, cosine, 0], [0, 0, 1]])
    r = {'x': rx, 'y': ry, 'z': rz}
    return np.dot(r[dim], point3d)


def convert_longlat(src, center, radius, out_height=OUT_HEIGHT, fov=220./180*PI, cam_mode=EQUIDISTANCE):
    out_width = out_height * 2
    focal = 2 * radius / fov
    ret = np.zeros((out_height, out_width, 3), np.uint8)
    src_size = src.shape
    for i in range(0, out_height):
        for j in range(0, out_width):
            # get points in sphere
            theta = j * 2 * PI / out_width - PI
            phi = i * PI / out_height - PI / 2
            x = np.cos(phi) * np.sin(theta)
            y = np.cos(phi) * np.cos(theta)
            z = np.sin(phi)
            # x, y, z = rotate3d((x, y, z), 'y', PI/2)

            # get points in circle
            theta2d = np.arctan2(z, x)
            phi2d = np.arctan2(np.sqrt(x * x + z * z), y)

            if cam_mode == 0:
                p = focal * phi2d
            elif cam_mode == 1:
                pass
            x1 = int(center[0] + p * np.cos(theta2d))
            y1 = int(center[1] + p * np.sin(theta2d))

            # map
            if 0 < x1 < src_size[0] and 0 < y1 < src_size[1]:
                ret[i][j] = src[y1][x1]
    return ret


# main
src = cv2.imread('fish2sphere220.jpg')
size = src.shape
res = convert_longlat(src, (size[0]/2, size[1]/2), size[0]/2)
cv2.imshow('res', res)
cv2.waitKey(0)

