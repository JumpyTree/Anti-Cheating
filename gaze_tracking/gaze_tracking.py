from __future__ import division
import os
import cv2
import dlib
import numpy as np

from .eye import Eye
from .calibration import Calibration
import time
import math
from queue import Queue

class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()
        self.rectangle_shape = None
        self.left_pupil = None
        self.right_pupil = None
        self.left_gaze = None
        self.right_gaze = None
        self.anomaly_queue_log = Queue()
        self.anomaly_queue_log2 = Queue()
        self.debug_mode = True

        self.pupil_positions = []
        self.logged_saccades = set()

        self.avg_horizontal_ratio = 0
        self.avg_vertical_ratio = 0
        self.avg_pupil_left_coords = (0, 0)
        self.avg_pupil_right_coords = (0, 0)
        self.avg_head_pose_angle = [0, 0, 0, 0, 0, 0, 0, 0]

        # Initialize tracking variables
        self.start_time = time.time()
        self.end_time = None
        self.previous_time = None
        self.num_frames = None
        self.delay_start = 5
        self.sampling_rate = None
        self.average_time_interval = 1  # Average time interval in seconds, set lower than 1 for real env

        # Initialize variables for deviation thresholds
        self._reset_averages()
        self.horizontal_ratio_deviation_threshold = 0.2
        self.vertical_ratio_deviation_threshold = 0.2
        self.pupil_coords_deviation_threshold = 150
        self.head_pose_angle_deviation_threshold = 100
        self.sampling_rate = None
        self.saccade_threshold = 37

        self.image_points_2d = None
        self.image_points_3d = None
        self.model_points = None

        self.b1 = None
        self.b2 = None
        self.b3 = None
        self.b4 = None
        self.b11 = None
        self.b12 = None
        self.b13 = None
        self.b14 = None

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # _predictor is used to get facial landmarks of a given face
        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @staticmethod
    def draw_line(frame, a, b, color=(255, 255, 0)):
        cv2.line(frame, a, b, color, 10)

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    def _reset_averages(self):
        # Reset average values and recorded data
        self.anomaly_queue_log.queue.clear()
        self.avg_horizontal_ratio = 0
        self.avg_vertical_ratio = 0
        self.avg_pupil_left_coords = (0, 0)
        self.avg_pupil_right_coords = (0, 0)

        self.horizontal_ratio_values = []
        self.vertical_ratio_values = []
        self.pupil_left_coords_values = []
        self.pupil_right_coords_values = []
        self.head_pose_angle_values = []

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        faces = self._face_detector(frame, 0)
        size = frame.shape

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)

            self.image_points_2d = np.array([
                (landmarks.part(33).x, landmarks.part(33).y),  # Nose tip
                (landmarks.part(8).x, landmarks.part(8).y),  # Chin
                (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
                (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corne
                (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
                (landmarks.part(54).x, landmarks.part(54).y)  # Right mouth corner
            ], dtype="double")

            self.image_points_3d = np.array([
                (landmarks.part(33).x, landmarks.part(33).y, 0),  # Nose tip
                (landmarks.part(8).x, landmarks.part(8).y, 0),  # Chin
                (landmarks.part(36).x, landmarks.part(36).y, 0),  # Left eye left corner
                (landmarks.part(45).x, landmarks.part(45).y, 0),  # Right eye right corner
                (landmarks.part(48).x, landmarks.part(48).y, 0),  # Left Mouth corner
                (landmarks.part(54).x, landmarks.part(54).y, 0)  # Right mouth corner
            ], dtype="double")

            # 3D model points.
            self.model_points = np.array([
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye left corner
                (225.0, 170.0, -135.0),  # Right eye right corner
                (-150.0, -150.0, -125.0),  # Left Mouth corner
                (150.0, -150.0, -125.0)  # Right mouth corner
            ])

            Eye_ball_center_right = np.array([[-145.05], [-163.5], [-197.5]])
            Eye_ball_center_left = np.array([[145.05], [-163.5], [-197.5]])  # the center of the left eyeball as a vector.

            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )

            dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points,
                self.image_points_2d,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            left_pupil = self.pupil_left_coords()
            right_pupil = self.pupil_right_coords()

            # Transformation between image point to world point
            #_, transformation, _ = cv2.estimateAffine3D(self.image_points_3d, self.model_points)  # image to world transformation
            transformation = None

            try:
                if transformation is not None:  # if estimateAffine3D succeeded
                    # project left pupil image point into 3D world point
                    left_pupil_world_cord = transformation @ np.array([[left_pupil[0], left_pupil[1], 0, 1]]).T

                    # project right pupil image point into 3D world point
                    right_pupil_world_cord = transformation @ np.array([[right_pupil[0], right_pupil[1], 0, 1]]).T

                    # 3D gaze points (10 is arbitrary value denoting gaze distance)
                    left_gaze = Eye_ball_center_left + (left_pupil_world_cord - Eye_ball_center_left) * 10
                    right_gaze = Eye_ball_center_right + (right_pupil_world_cord - Eye_ball_center_right) * 10

                    # Project the 3D gaze directions onto the image plane.
                    (left_gaze_2d, _) = cv2.projectPoints((int(left_gaze[0]), int(left_gaze[1]), int(left_gaze[2])),
                                                          rotation_vector, translation_vector, camera_matrix,
                                                          dist_coeffs)
                    (right_gaze_2d, _) = cv2.projectPoints((int(right_gaze[0]), int(right_gaze[1]), int(right_gaze[2])),
                                                           rotation_vector, translation_vector, camera_matrix,
                                                           dist_coeffs)

                    # project 3D head pose into the image plane
                    (left_head_pose, _) = cv2.projectPoints(
                        (int(left_pupil_world_cord[0]), int(left_pupil_world_cord[1]), int(40)),
                        rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                    (right_head_pose, _) = cv2.projectPoints(
                        (int(right_pupil_world_cord[0]), int(right_pupil_world_cord[1]), int(40)),
                        rotation_vector, translation_vector, camera_matrix, dist_coeffs)

                    # correct gaze for head rotation
                    left_gaze_direction = left_pupil + (left_gaze_2d[0][0] - left_pupil) - (
                            left_head_pose[0][0] - left_pupil)
                    right_gaze_direction = right_pupil + (right_gaze_2d[0][0] - right_pupil) - (
                            right_head_pose[0][0] - right_pupil)

                    self.left_pupil = (int(left_pupil[0]), int(left_pupil[1]))
                    self.left_gaze = (int(left_gaze_direction[0]), int(left_gaze_direction[1]))
                    self.right_pupil = (int(right_pupil[0]), int(right_pupil[1]))
                    self.right_gaze = (int(right_gaze_direction[0]), int(right_gaze_direction[1]))
                #else:
                #    print("No Transformation")
            except TypeError as e:
                print("Transformation error occurred:", e)


            (self.b1, jacobian) = cv2.projectPoints(np.array([(350.0, 270.0, 0.0)]), rotation_vector,
                                                    translation_vector, camera_matrix, dist_coeffs)
            (self.b2, jacobian) = cv2.projectPoints(np.array([(-350.0, -270.0, 0.0)]), rotation_vector,
                                                    translation_vector, camera_matrix, dist_coeffs)
            (self.b3, jacobian) = cv2.projectPoints(np.array([(-350.0, 270, 0.0)]), rotation_vector, translation_vector,
                                                    camera_matrix, dist_coeffs)
            (self.b4, jacobian) = cv2.projectPoints(np.array([(350.0, -270.0, 0.0)]), rotation_vector,
                                                    translation_vector, camera_matrix, dist_coeffs)

            (self.b11, jacobian) = cv2.projectPoints(np.array([(450.0, 350.0, 400.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)
            (self.b12, jacobian) = cv2.projectPoints(np.array([(-450.0, -350.0, 400.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)
            (self.b13, jacobian) = cv2.projectPoints(np.array([(-450.0, 350, 400.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)
            (self.b14, jacobian) = cv2.projectPoints(np.array([(450.0, -350.0, 400.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)

            self.b1 = (int(self.b1[0][0][0]), int(self.b1[0][0][1]))
            self.b2 = (int(self.b2[0][0][0]), int(self.b2[0][0][1]))
            self.b3 = (int(self.b3[0][0][0]), int(self.b3[0][0][1]))
            self.b4 = (int(self.b4[0][0][0]), int(self.b4[0][0][1]))

            self.b11 = (int(self.b11[0][0][0]), int(self.b11[0][0][1]))
            self.b12 = (int(self.b12[0][0][0]), int(self.b12[0][0][1]))
            self.b13 = (int(self.b13[0][0][0]), int(self.b13[0][0][1]))
            self.b14 = (int(self.b14[0][0][0]), int(self.b14[0][0][1]))

            newRect = dlib.rectangle(int(faces[0].left()), int(faces[0].top()), int(faces[0].right()),
                                     int(faces[0].bottom()))
            # Find face landmarks by providing rectangle for each face
            self.rectangle_shape = self._predictor(frame, newRect)

            self._update_averages()

        except IndexError:
            self.eye_left = None
            self.eye_right = None

    def _update_averages(self):
        """Update the average values of horizontal and vertical ratios, pupil coordinates, and head pose angle"""
        if self.start_time is None:
            self.start_time = time.time()

        if self.previous_time is None:
            self.previous_time = time.time()

        if self.num_frames is None:
            self.num_frames = 1
        else:
            self.num_frames += 1

        elapsed_time = time.time() - self.previous_time

        if elapsed_time != 0:
            self.sampling_rate = 1 / elapsed_time
        else:
            self.sampling_rate = 1

        self.previous_time = time.time()

        if True:
            pupil_left_coords = self.pupil_left_coords()
            pupil_right_coords = self.pupil_right_coords()
            horizontal_ratio = self.horizontal_ratio()
            vertical_ratio = self.vertical_ratio()
            head_pose_angle = self.head_pose_angle()

            if pupil_left_coords is not None and pupil_right_coords is not None:
                self.detect_saccades()

            if pupil_left_coords is not None:
                avg_pupil_left_x = (self.avg_pupil_left_coords[0] * self.num_frames + pupil_left_coords[0]) / (
                        self.num_frames + 1)
                avg_pupil_left_y = (self.avg_pupil_left_coords[1] * self.num_frames + pupil_left_coords[1]) / (
                        self.num_frames + 1)
                self.avg_pupil_left_coords = (avg_pupil_left_x, avg_pupil_left_y)
                deviation_left_x = abs(pupil_left_coords[0] - self.avg_pupil_left_coords[0])
                deviation_left_y = abs(pupil_left_coords[1] - self.avg_pupil_left_coords[1])
                if deviation_left_x > self.pupil_coords_deviation_threshold or deviation_left_y > self.pupil_coords_deviation_threshold:
                    deviation_info = {
                        'frame': self.num_frames,
                        'timestamp': time.ctime(time.time()),
                        'case': "pupil position",
                        'type': "deviation",
                        'info': {
                            'avg_left_pupil_pos': self.avg_pupil_left_coords,
                            'pupil_left_coords': pupil_left_coords,
                            'deviation_left_x': deviation_left_x,
                            'deviation_left_y': deviation_left_y
                        }
                    }
                    self.anomaly_queue_log.put(deviation_info)

            if pupil_right_coords is not None:
                avg_pupil_right_x = (self.avg_pupil_right_coords[0] * self.num_frames + pupil_right_coords[0]) / (
                            self.num_frames + 1)
                avg_pupil_right_y = (self.avg_pupil_right_coords[1] * self.num_frames + pupil_right_coords[1]) / (
                            self.num_frames + 1)
                self.avg_pupil_right_coords = (avg_pupil_right_x, avg_pupil_right_y)
                deviation_right_x = abs(pupil_right_coords[0] - self.avg_pupil_right_coords[0])
                deviation_right_y = abs(pupil_right_coords[1] - self.avg_pupil_right_coords[1])
                if deviation_right_x > self.pupil_coords_deviation_threshold or deviation_right_y > self.pupil_coords_deviation_threshold:
                    deviation_info = {
                        'frame': self.num_frames,
                        'timestamp': time.ctime(time.time()),
                        'case': "pupil position",
                        'type': "deviation",
                        'info': {
                            'avg_pupil_right_coords': self.avg_pupil_right_coords,
                            'pupil_right_coords': pupil_right_coords,
                            'deviation_right_x': deviation_right_x,
                            'deviation_right_y': deviation_right_y
                        }
                    }
                    self.anomaly_queue_log.put(deviation_info)

            if horizontal_ratio is not None:
                self.avg_horizontal_ratio = (self.avg_horizontal_ratio * self.num_frames + horizontal_ratio) / (
                            self.num_frames + 1)
                deviation_horizontal = abs(horizontal_ratio - self.avg_horizontal_ratio)
                if deviation_horizontal > self.horizontal_ratio_deviation_threshold:
                    deviation_info = {
                        'frame': self.num_frames,
                        'timestamp': time.ctime(time.time()),
                        'case': "horizontal ratio",
                        'type': "deviation",
                        'info': {
                            'avg_horizontal_ratio': self.avg_horizontal_ratio,
                            'horizontal_ratio': horizontal_ratio,
                            'deviation_horizontal': deviation_horizontal
                        }
                    }
                    self.anomaly_queue_log.put(deviation_info)

            if vertical_ratio is not None:
                self.avg_vertical_ratio = (self.avg_vertical_ratio * self.num_frames + vertical_ratio) / (self.num_frames + 1)
                deviation_vertical = abs(vertical_ratio - self.avg_vertical_ratio)
                if deviation_vertical > self.vertical_ratio_deviation_threshold:
                    deviation_info = {
                        'frame': self.num_frames,
                        'timestamp': time.ctime(time.time()),
                        'case': "vertical ratio",
                        'type': "deviation",
                        'info': {
                            'avg_vertical_ratio': self.avg_vertical_ratio,
                            'vertical_ratio': vertical_ratio,
                            'deviation_vertical': deviation_vertical
                        }
                    }
                    self.anomaly_queue_log.put(deviation_info)

            if head_pose_angle is not None:
                head_pose_angle_deviation = False
                for i in range(8):
                    self.avg_head_pose_angle[i] = (self.avg_head_pose_angle[i] * self.num_frames + head_pose_angle[i]) / (
                            self.num_frames + 1)
                    deviation_angle = abs(head_pose_angle[i] - self.avg_head_pose_angle[i])
                    if deviation_angle > self.head_pose_angle_deviation_threshold:
                        deviation_info = {
                            'frame': self.num_frames,
                            'timestamp': time.ctime(time.time()),
                            'case': "head pose angle",
                            'type': "deviation",
                            'info': {
                                'avg_head_pose_angle': self.avg_head_pose_angle,
                                'head_pose_angle': head_pose_angle,
                                'deviation_angle': deviation_angle
                            }
                        }
                        head_pose_angle_deviation = True
                if head_pose_angle_deviation:
                    self.anomaly_queue_log.put(deviation_info)

        self.anomaly_queue_log2 = self.anomaly_queue_log

    def calculate_velocities(self, pupil_positions_array):
        x_positions = pupil_positions_array[:, 0]
        y_positions = pupil_positions_array[:, 1]
        dx = np.diff(x_positions)
        dy = np.diff(y_positions)
        velocities = np.sqrt(dx ** 2 + dy ** 2) * self.sampling_rate
        return velocities

    def detect_saccades(self):
        middle_coordinate = ((self.pupil_left_coords()[0] + self.pupil_right_coords()[0]) / 2,
                             (self.pupil_left_coords()[1] + self.pupil_right_coords()[1]) / 2)
        self.pupil_positions.append(middle_coordinate)

        pupil_positions_array = np.array(self.pupil_positions)
        velocities = self.calculate_velocities(pupil_positions_array)
        saccade_indices = np.where(velocities > self.saccade_threshold)[0]

        new_saccades = set(saccade_indices) - self.logged_saccades
        self.logged_saccades.update(new_saccades)

        for saccade_index in new_saccades:
            saccade_info = {
                'frame': self.num_frames,
                'timestamp': time.ctime(time.time()),
                'case': "pupil",
                'type': "saccade",
                'info': {
                    'middle_coordinate': middle_coordinate,
                    'saccade_threshold': self.saccade_threshold,
                }
            }

            self.anomaly_queue_log.put(saccade_info)

        return list(new_saccades)

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def head_pose_angle(self):
        def calculate_line_length(x1, y1, x2, y2):
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            return length

        def calculate_line_angle(x1, y1, x2, y2):
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
            return angle

        # Calculate length and angle for each line
        length_1 = calculate_line_length(self.b11[0], self.b11[1], self.b1[0], self.b1[1])
        angle_1 = calculate_line_angle(self.b11[0], self.b11[1], self.b1[0], self.b1[1])

        length_2 = calculate_line_length(self.b12[0], self.b12[1], self.b2[0], self.b2[1])
        angle_2 = calculate_line_angle(self.b12[0], self.b12[1], self.b2[0], self.b2[1])

        length_3 = calculate_line_length(self.b13[0], self.b13[1], self.b3[0], self.b3[1])
        angle_3 = calculate_line_angle(self.b13[0], self.b13[1], self.b3[0], self.b3[1])

        length_4 = calculate_line_length(self.b14[0], self.b14[1], self.b4[0], self.b4[1])
        angle_4 = calculate_line_angle(self.b14[0], self.b14[1], self.b4[0], self.b4[1])

        return [length_1, angle_1, length_2, angle_2, length_3, angle_3, length_4, angle_4]

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.5

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.7

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def toggle_debug(self):
        if not self.debug_mode:
            self.debug_mode = True
        else:
            self.debug_mode = False

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located and self.debug_mode:

            # Mark Pupils
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()

            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

            # Draw Landmarks
            for p in self.rectangle_shape.parts():
                cv2.circle(frame, (p.x, p.y), 2, (0, 255, 0), -1)

            # Inner sides of the box
            self.draw_line(frame, self.b1, self.b3, color=(0, 255, 0))  # Top side
            self.draw_line(frame, self.b3, self.b2, color=(0, 255, 0))  # Left side
            self.draw_line(frame, self.b2, self.b4, color=(0, 255, 0))  # Bottom side
            self.draw_line(frame, self.b4, self.b1, color=(0, 255, 0))  # Right side

            # Outer sides of the box
            self.draw_line(frame, self.b11, self.b13, color=(255, 0, 0))  # Top side
            self.draw_line(frame, self.b13, self.b12, color=(255, 0, 0))  # Left side
            self.draw_line(frame, self.b12, self.b14, color=(255, 0, 0))  # Bottom side
            self.draw_line(frame, self.b14, self.b11, color=(255, 0, 0))  # Right side

            # Middle sides of the box
            self.draw_line(frame, self.b11, self.b1, color=(0, 0, 255))  # Upper Right
            self.draw_line(frame, self.b13, self.b3, color=(0, 0, 255))  # Upper Left
            self.draw_line(frame, self.b12, self.b2, color=(0, 0, 255))  # Lower Left
            self.draw_line(frame, self.b14, self.b4, color=(0, 0, 255))  # Lower Right

            self.draw_line(frame, self.b12, self.b2, color=(0, 0, 255))  # Lower Left
            self.draw_line(frame, self.b14, self.b4, color=(0, 0, 255))  # Lower Right
            # Draw gaze lines on the frame
            cv2.line(frame, self.left_pupil, self.left_gaze, (0, 0, 255), 2)

            cv2.line(frame, self.right_pupil, self.right_gaze, (0, 0, 255), 2)
        return frame
