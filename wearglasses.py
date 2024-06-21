import os
import sys
import argparse
import numpy as np
import cv2
import face_recognition
import math
import dlib
from PIL import Image, ImageFile
import random

__version__ = '0.3.0'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'glasses')
# IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
GLASSES1_IMAGE_PATH = os.path.join(IMAGE_DIR, 'glasses1.png')
GLASSES2_IMAGE_PATH = os.path.join(IMAGE_DIR, 'glasses2.png')
GLASSES3_IMAGE_PATH = os.path.join(IMAGE_DIR, 'glasses3.png')
GLASSES4_IMAGE_PATH = os.path.join(IMAGE_DIR, 'glasses4.png')
GLASSES5_IMAGE_PATH = os.path.join(IMAGE_DIR, 'glasses5.png')
GLASSES6_IMAGE_PATH = os.path.join(IMAGE_DIR, 'glasses6.png')
GLASSES7_IMAGE_PATH = os.path.join(IMAGE_DIR, 'glasses7.png')
GLASSES8_IMAGE_PATH = os.path.join(IMAGE_DIR, 'glasses8.png')
GLASSES9_IMAGE_PATH = os.path.join(IMAGE_DIR, 'glasses9.png')

def rect_to_bbox(rect):
    """获得人脸矩形的坐标信息"""
    # print(rect)
    x = rect[3]
    y = rect[0]
    w = rect[1] - x
    h = rect[2] - y
    return (x, y, w, h)

predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
def face_alignment(faces):
    # 预测关键点
    
    faces_aligned = []
    for face in faces:

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
        shape = predictor(face_gray, rec)
        #shape = predictor(np.uint8(face), rec)
        # left eye, right eye, nose, left mouth, right mouth
        order = [36, 45, 30, 48, 54]
        for j in order:
            x = shape.part(j).x
            y = shape.part(j).y
        # 计算两眼的中心坐标
        eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2, (shape.part(36).y + shape.part(45).y) * 1. / 2)
        dx = (shape.part(45).x - shape.part(36).x)
        dy = (shape.part(45).y - shape.part(36).y)
        # 计算角度
        angle = math.atan2(dy, dx) * 180. / math.pi
        # 计算仿射矩阵
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
        # 进行仿射变换，即旋转
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))
        faces_aligned.append(RotImg)
    return faces_aligned

def cli(pic_path ,save_pic_path):
    parser = argparse.ArgumentParser(description='Wear a face mask in the given picture.')
    # parser.add_argument('pic_path', default='/Users/wuhao/lab/wear-a-mask/spider/new_lfw/Aaron_Tippin/Aaron_Tippin_0001.jpg',help='Picture path.')
    # parser.add_argument('--show', action='store_true', help='Whether show picture with mask or not.')
    parser.add_argument('--model', default='hog', choices=['hog', 'cnn'], help='Which face detection model to use.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--black', action='store_true', help='Wear black mask')
    group.add_argument('--blue', action='store_true', help='Wear blue mask')
    group.add_argument('--red', action='store_true', help='Wear red mask')
    args = parser.parse_args()

    if not os.path.exists(pic_path):
        print(f'Picture {pic_path} not exists.')
        sys.exit(1)
    mask_path = random.choice([GLASSES1_IMAGE_PATH, GLASSES3_IMAGE_PATH,
                               GLASSES4_IMAGE_PATH, GLASSES5_IMAGE_PATH,
                               GLASSES6_IMAGE_PATH, GLASSES7_IMAGE_PATH,
                               GLASSES8_IMAGE_PATH, GLASSES9_IMAGE_PATH])
    # mask_path = random.choice([GLASSES3_IMAGE_PATH])

    unmasked_paths = FaceMasker(pic_path, mask_path, True, 'cnn',save_pic_path).mask()
    return unmasked_paths

class FaceMasker:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

    def __init__(self, face_path, mask_path, show=False, model='cnn',save_path = ''):
        self.face_path = face_path
        self.mask_path = mask_path
        self.save_path = save_path
        self.show = show
        self.model = model
        self._face_img: ImageFile = None
        self._mask_img: ImageFile = None

    def mask(self):
        face_image_np = face_recognition.load_image_file(self.face_path)
        face_locations = face_recognition.face_locations(face_image_np, model=self.model)
        face_landmarks = face_recognition.face_landmarks(face_image_np)
        self._face_img = Image.fromarray(face_image_np)
        self._mask_img = Image.open(self.mask_path)

        found_face = False
        for face_landmark in face_landmarks:
            # check whether facial features meet requirement
            skip = False
            for facial_feature in self.KEY_FACIAL_FEATURES:
                if facial_feature not in face_landmark:
                    skip = True
                    break
            if skip:
                continue

            # mask face
            #found_face = True
            #self._mask_face(face_landmark)


            # image_cv = np.array(self._face_img)
            # for key in face_landmark.keys():
            #     for p in face_landmark[key]:
            #         cv2.circle(image_cv,p, 1,(0,255,255),1,1)
            # cv2.imshow('show', image_cv)
            # cv2.waitKey(0)
    

            found_face = True
            self._mask_face(face_landmark)

        unmasked_paths = []

        if found_face:
            # align
            src_faces = []
            src_face_num = 0
            with_mask_face = np.asarray(self._face_img)
            for (i, rect) in enumerate(face_locations):
                src_face_num = src_face_num + 1
                (x, y, w, h) = rect_to_bbox(rect)
                detect_face = with_mask_face[y:y + h, x:x + w]
                # src_faces.append(detect_face)
                src_faces.append(with_mask_face)
            # 人脸对齐操作并保存
            # faces_aligned = face_alignment(src_faces)
            faces_aligned = src_faces
            face_num = 0
            for faces in faces_aligned:
                face_num = face_num + 1
                faces = cv2.cvtColor(faces, cv2.COLOR_RGBA2BGR)
                cv2.imwrite(self.save_path, faces)
        else:
            #在这里记录没有裁的图片
            print('Found no face.' + self.save_path)
            unmasked_paths.append(self.save_path)

        return unmasked_paths

    def _mask_face(self, face_landmark: dict):
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        nose_v = np.array(nose_point)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]
        
        left_eyebrow = face_landmark['left_eyebrow']
        eyebrow_left_point = left_eyebrow[0]
        
        right_eyebrow = face_landmark['right_eyebrow']
        eyebrow_right_point = right_eyebrow[-1]

        # split mask and resize
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.2
        height_ratio = 1.5
        if self.mask_path == GLASSES2_IMAGE_PATH:
            height_ratio = 2.5

        # new_height = int(np.linalg.norm(nose_v - chin_bottom_v))
        new_height = int(height_ratio * self.get_distance_from_point_to_line(nose_point, eyebrow_left_point, eyebrow_right_point))


        img_cv = np.array(self._face_img)

        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(eyebrow_left_point, nose_point, chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        if  mask_left_width > 0 and new_height > 0:
          mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(eyebrow_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        if  mask_right_width > 0 and new_height> 0:
          mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        # mask_img = Image.new('RGB', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask
        x = chin_bottom_point[0] - nose_point[0]
        y = chin_bottom_point[1] - nose_point[1]
        angle = np.degrees(np.arctan2(x, y))
        rotated_mask_img = mask_img.rotate(angle, expand=True)
        # rotated_mask_img.save('after_rotate.jpg')
        
        
        
        # calculate mask location
        # center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        # center_y = (nose_point[1] + chin_bottom_point[1]) // 2
        center_x = nose_bridge[0][0] 
        center_y = nose_bridge[0][1] 

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # add mask
        self._face_img.paste(rotated_mask_img, (box_x, box_y), rotated_mask_img)

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


if __name__ == '__main__':
    import os
    import glob
    dataset_path = './imgs'
    unmasked_paths=[]
    files = glob.glob(os.path.join(dataset_path, '*'))
    for i, file in enumerate(files):
        print("{}/{}".format(i, len(files)))
        # deal
        imgpath = file
        save_imgpath = os.path.join('results', os.path.basename(file))
        if not os.path.exists(os.path.dirname(save_imgpath)):
            os.makedirs(os.path.dirname(save_imgpath), exist_ok=True)
        unmasked_paths = cli(imgpath,save_imgpath)
