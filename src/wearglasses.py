import os
import glob
import numpy as np
import cv2
import face_recognition
from PIL import Image, ImageFile

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'something_templetes','glasses')
GLASSES2_IMAGE_PATH = os.path.join(IMAGE_DIR, 'glasses2.png')
GLASS_PATH_LIST = glob.glob(os.path.join(IMAGE_DIR, '*'))

class FaceGlasser:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin','left_eyebrow','right_eyebrow')

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

            found_face = True
            self._glasses_face(face_landmark)

        if found_face:
            with_mask_face = np.asarray(self._face_img)
            faces = cv2.cvtColor(with_mask_face, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(self.save_path, faces)
        else:
            print('Found no face.' + self.save_path)

    def _glasses_face(self, face_landmark: dict):
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        
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

        new_height = int(height_ratio * self.get_distance_from_point_to_line(nose_point, eyebrow_left_point, eyebrow_right_point))

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

        # calculate mask location
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


