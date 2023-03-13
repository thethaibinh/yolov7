import argparse
import time
from pathlib import Path
import yaml
import math
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf2_ros, tf2_geometry_msgs
from geometry_msgs.msg import Pose, Point, PointStamped, PoseStamped
from visualization_msgs.msg import Marker


class FSDNode:
    def __init__(self, config):
        print("Initializing free space partitioner...")
        print("Preparing model...")

        # Arguments and attributes
        self.config = config
        self.goal = np.float32(self.config['goal'])
        self.checking_depth = np.float32(self.config['checking_depth'])
        self.planning_depth = np.float32(self.config['planning_depth'])
        self.sfc_minimum_width = np.float32(self.config['sfc_minimum_width'])
        self.sfc_minimum_height = np.float32(self.config['sfc_minimum_height'])
        self.source, self.weights, self.view_img, self.save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
        self.save_img = not opt.nosave and not self.source.endswith('.txt')  # save inference images
        self.count = 0
        self.best_endpoint = [0, 0, 0] # Initialising the best direction endpoint
        self.last_generated_time = time.time()
        self.steering_sent = False
        self.pose_transformed = PoseStamped()

        # Directories
        self.save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        torch.cuda.empty_cache()
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=self.stride)  # check img_size

        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.img_size = imgsz
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.width = 320
        self.height = 240
        fov = 90.0
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.fx = (self.width / 2.0) / math.tan((math.pi * fov / 180.0) / 2.0)
        self.fy = (self.height / 2.0) / math.tan((math.pi * fov / 180.0) / 2.0)
        # depth scale for 32FC1 is 100, 16UC1 is scaled up by 65535
        self.depth_scale = 100.0 / 65535.0

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1

        print("Model ready!")

        print("Preparing ROS node for FSD planner...")
        rospy.init_node('fsd', anonymous=False)
        self.cv_bridge = CvBridge()
        # Observation subscribers
        self.img_sub = rospy.Subscriber(self.source, Image, self.img_callback,
                                        queue_size=1, tcp_nodelay=True)
        self.trajectory_pub = rospy.Publisher("/trajectory", Pose, queue_size=1)
        self.visual_pub = rospy.Publisher("/visualization", Marker, queue_size=10)
        # Initialising transform buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        print("Initialization completed!")

    def get_goal_in_camera_frame(self):
        pointstamp = PointStamped()
        pointstamp.header.frame_id = 'world'
        pointstamp.header.stamp = rospy.Time.now()
        pointstamp.point.x = self.goal[0]
        pointstamp.point.y = self.goal[1]
        pointstamp.point.z = self.goal[2]
        try:
            transform = self.tf_buffer.lookup_transform('vehicle',
                                                        # source frame:
                                                        pointstamp.header.frame_id,
                                                        # get the tf at the time the pose was valid
                                                        pointstamp.header.stamp,
                                                        # take the latest transform otherwise throw
                                                        rospy.Duration(1))
            pose_transformed = tf2_geometry_msgs.do_transform_point(pointstamp, transform)
            return [-pose_transformed.point.y, -pose_transformed.point.z, pose_transformed.point.x]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(e)
            return [0, 0, 2]

    def pixel_in_bounding_box(self, pixel, bb):
        left    = int(bb[0])
        top     = int(bb[1])
        right   = int(bb[2])
        bottom  = int(bb[3])
        x = pixel[0]
        y = pixel[1]
        if (x - left) * (x - right) > 0:
            return False
        if (y - top) * (y - bottom) > 0:
            return False
        return True

    def deproject_pixel_to_point(self, pixel, depth):
        return [(pixel[0] - self.cx) * depth / self.fx, (pixel[1] - self.cy) * depth / self.fy, depth]

    def closest_pixel_in_bounding_box(self, pixel, bb):
        left    = int(bb[0])
        top     = int(bb[1])
        right   = int(bb[2])
        bottom  = int(bb[3])
        x = pixel[0]
        y = pixel[1]
        # this function implements the solution described in https://math.stackexchange.com/a/356813
        if y < top:
            if x < left:
                return [left, top]
            elif (x - left) * (x - right) <= 0:
                return [x, top]
            elif x > right:
                return [right, top]
        elif (y - top) * (y - bottom) <=0:
            if x < left:
                return [left, y]
            elif x > right:
                return [right, y]
        elif y > bottom:
            if x < left:
                return [left, bottom]
            elif (x - left) * (x - right) <= 0:
                return [x, bottom]
            elif x > right:
                return [right, bottom]

    def project_point_to_pixel(self, point):
        pixel_x = point[0] * self.fx / abs(point[2]) + self.cx
        pixel_y = point[1] * self.fy / abs(point[2]) + self.cy
        return [pixel_x, pixel_y]

    def publish_endpoint(self, feasible, endpoint, img_data):
        traj_in_vehicle_frame = PoseStamped()
        traj_in_vehicle_frame.header.frame_id = 'vehicle'
        traj_in_vehicle_frame.header.stamp = rospy.Time.now()
        # rotating from camera frame to vehicle frame
        traj_in_vehicle_frame.pose.position.x = endpoint[2]
        traj_in_vehicle_frame.pose.position.y = -endpoint[0]
        traj_in_vehicle_frame.pose.position.z = -endpoint[1]
        try:
            transform = self.tf_buffer.lookup_transform('world',
                                                        # source frame:
                                                        traj_in_vehicle_frame.header.frame_id,
                                                        # get the tf at the time the pose was valid
                                                        traj_in_vehicle_frame.header.stamp,
                                                        # take the latest transform otherwise throw
                                                        rospy.Duration(1))

            # For convenience purposes, orientation.x = 1 means we are sending steering commands
            # we use orientation.z for steering value
            if feasible:
                self.last_generated_time = time.time()
                self.steering_sent = False
                # Transforming endpoint from vehicle frame to world frame
                self.pose_transformed = tf2_geometry_msgs.do_transform_pose(traj_in_vehicle_frame, transform)
                self.pose_transformed.pose.orientation.w = 1
                self.pose_transformed.pose.orientation.x = 0
                self.pose_transformed.pose.orientation.z = 0
                self.trajectory_pub.publish(self.pose_transformed.pose)
                self.visualise_endpoint(endpoint)
                return
            if (time.time() - self.last_generated_time) < 1 or self.steering_sent:
                return

            self.steering_sent = True
            steer_value = self.get_steering(img_data)
            self.pose_transformed.pose.orientation.w = 1
            self.pose_transformed.pose.orientation.x = 1
            self.pose_transformed.pose.orientation.z = steer_value
            self.trajectory_pub.publish(self.pose_transformed.pose)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as err:
            print(err)

    def get_steering(self, img_data):
        '''
        get steering value from current depth image
        '''
        depth_mat = self.cv_bridge.imgmsg_to_cv2(img_data,
                                                 desired_encoding='passthrough') * self.depth_scale
        index = np.argmin(depth_mat, axis=None)
        x_index = index % self.width
        if x_index < self.cx:
            return 1
        else:
            return -1

    def visualise_endpoint(self, endpoint):
        trajectory = Marker()
        trajectory.header.frame_id = "vehicle"
        trajectory.type = trajectory.LINE_STRIP
        trajectory.action = trajectory.ADD
        trajectory.id = 3
        trajectory.ns = "visualization"

        # trajectory scale
        trajectory.scale.x = 0.02
        trajectory.scale.y = 0.02
        trajectory.scale.z = 0.02

        # trajectory color
        trajectory.color.a = 1.0
        trajectory.color.r = 0.0
        trajectory.color.g = 0.0
        trajectory.color.b = 1.0

        # trajectory orientaiton
        trajectory.pose.orientation.x = 0.0
        trajectory.pose.orientation.y = 0.0
        trajectory.pose.orientation.z = 0.0
        trajectory.pose.orientation.w = 1.0

        # trajectory position
        trajectory.pose.position.x = 0.0
        trajectory.pose.position.y = 0.0
        trajectory.pose.position.z = 0.0

        # trajectory line points
        trajectory.points = []
        # first point
        p = Point()
        p.x = 0.0
        p.y = 0.0
        p.z = 0.0
        trajectory.points.append(p)
        # second point
        p1 = Point()
        # rotating from camera frame to vehicle frame
        p1.x = endpoint[2]
        p1.y = -endpoint[0]
        p1.z = -endpoint[1]
        trajectory.points.append(p1)
        self.visual_pub.publish(trajectory)

    def visualise_SFC(self, bb):
        self.visualise_edge(0, bb, 'g')
        self.visualise_bb(1, bb, 'g')
        self.visualise_bb(2, [0, 0, self.width, self.height], 'r')

    def visualise_edge(self, id, bb, color):
        # parameters for visualization
        pyramids_edges = Marker()
        pyramids_edges.header.frame_id = "vehicle"
        pyramids_edges.type = Marker.LINE_LIST
        pyramids_edges.action = Marker.ADD
        pyramids_edges.id = id
        pyramids_edges.ns = "visualization"

        # scale
        pyramids_edges.scale.x = 0.02
        pyramids_edges.scale.y = 0.02
        pyramids_edges.scale.z = 0.02

        # color
        pyramids_edges.color.a = 1.0
        if color == 'o':
            pyramids_edges.color.r = 1.0
            pyramids_edges.color.g = 1.0
            pyramids_edges.color.b = 0.0
        elif color == 'r':
            pyramids_edges.color.r = 1.0
            pyramids_edges.color.g = 0.0
            pyramids_edges.color.b = 0.0
        elif color == 'g':
            pyramids_edges.color.r = 0.0
            pyramids_edges.color.g = 1.0
            pyramids_edges.color.b = 0.0

        # orientation
        pyramids_edges.pose.orientation.x = 0.0
        pyramids_edges.pose.orientation.y = 0.0
        pyramids_edges.pose.orientation.z = 0.0
        pyramids_edges.pose.orientation.w = 1.0

        # position
        pyramids_edges.pose.position.x = 0.0
        pyramids_edges.pose.position.y = 0.0
        pyramids_edges.pose.position.z = 0.0

        # line points
        pyramids_edges.points = []

        # Publish safe flight corridor
        left    = int(bb[0])
        top     = int(bb[1])
        right   = int(bb[2])
        bottom  = int(bb[3])

        # left top
        p0 = Point()
        p0.x = p0.y = p0.z = 0.0
        pyramids_edges.points.append(p0)
        left_top = self.deproject_pixel_to_point([left, top], self.checking_depth)
        p1 = Point()
        # rotating from camera frame to vehicle frame
        p1.x = left_top[2]
        p1.y = -left_top[0]
        p1.z = -left_top[1]
        pyramids_edges.points.append(p1)

        # right top
        pyramids_edges.points.append(p0)
        right_top = self.deproject_pixel_to_point([right, top], self.checking_depth)
        p3 = Point()
        # rotating from camera frame to vehicle frame
        p3.x = right_top[2]
        p3.y = -right_top[0]
        p3.z = -right_top[1]
        pyramids_edges.points.append(p3)

        # right bottom
        pyramids_edges.points.append(p0)
        right_bottom = self.deproject_pixel_to_point([right, bottom], self.checking_depth)
        p5 = Point()
        # rotating from camera frame to vehicle frame
        p5.x = right_bottom[2]
        p5.y = -right_bottom[0]
        p5.z = -right_bottom[1]
        pyramids_edges.points.append(p5)

        # left bottom
        pyramids_edges.points.append(p0)
        left_bottom = self.deproject_pixel_to_point([left, bottom], self.checking_depth)
        p7 = Point()
        # rotating from camera frame to vehicle frame
        p7.x = left_bottom[2]
        p7.y = -left_bottom[0]
        p7.z = -left_bottom[1]
        pyramids_edges.points.append(p7)

        # publishing the edge
        self.visual_pub.publish(pyramids_edges)

    def visualise_bb(self, id, bb, color):
        # parameters for visualization
        pyramids_bases = Marker()
        pyramids_bases.header.frame_id = "vehicle"
        pyramids_bases.type = Marker.LINE_STRIP
        pyramids_bases.action = Marker.ADD
        pyramids_bases.id = id
        pyramids_bases.ns = "visualization"

        # scale
        pyramids_bases.scale.x = 0.02
        pyramids_bases.scale.y = 0.02
        pyramids_bases.scale.z = 0.02

        # color
        pyramids_bases.color.a = 1.0
        if color == 'o':
            pyramids_bases.color.r = 1.0
            pyramids_bases.color.g = 1.0
            pyramids_bases.color.b = 0.0
        elif color == 'r':
            pyramids_bases.color.r = 1.0
            pyramids_bases.color.g = 0.0
            pyramids_bases.color.b = 0.0
        elif color == 'g':
            pyramids_bases.color.r = 0.0
            pyramids_bases.color.g = 1.0
            pyramids_bases.color.b = 0.0

        # orientation
        pyramids_bases.pose.orientation.x = 0.0
        pyramids_bases.pose.orientation.y = 0.0
        pyramids_bases.pose.orientation.z = 0.0
        pyramids_bases.pose.orientation.w = 1.0

        # position
        pyramids_bases.pose.position.x = 0.0
        pyramids_bases.pose.position.y = 0.0
        pyramids_bases.pose.position.z = 0.0

        # line points
        pyramids_bases.points = []

        # Publish safe flight corridor
        left    = int(bb[0])
        top     = int(bb[1])
        right   = int(bb[2])
        bottom  = int(bb[3])

        # left top
        left_top = self.deproject_pixel_to_point([left, top], self.checking_depth)
        p1 = Point()
        # rotating from camera frame to vehicle frame
        p1.x = left_top[2]
        p1.y = -left_top[0]
        p1.z = -left_top[1]
        pyramids_bases.points.append(p1)

        # right top
        right_top = self.deproject_pixel_to_point([right, top], self.checking_depth)
        p3 = Point()
        # rotating from camera frame to vehicle frame
        p3.x = right_top[2]
        p3.y = -right_top[0]
        p3.z = -right_top[1]
        pyramids_bases.points.append(p3)

        # right bottom
        right_bottom = self.deproject_pixel_to_point([right, bottom], self.checking_depth)
        p5 = Point()
        # rotating from camera frame to vehicle frame
        p5.x = right_bottom[2]
        p5.y = -right_bottom[0]
        p5.z = -right_bottom[1]
        pyramids_bases.points.append(p5)

        # left bottom
        left_bottom = self.deproject_pixel_to_point([left, bottom], self.checking_depth)
        p7 = Point()
        # rotating from camera frame to vehicle frame
        p7.x = left_bottom[2]
        p7.y = -left_bottom[0]
        p7.z = -left_bottom[1]
        pyramids_bases.points.append(p7)

        # left top closing the base
        p8 = Point()
        # rotating from camera frame to vehicle frame
        p8.x = left_top[2]
        p8.y = -left_top[0]
        p8.z = -left_top[1]
        pyramids_bases.points.append(p8)

        # publishing the SFC
        self.visual_pub.publish(pyramids_bases)

    def verify_SFC(self, bb, img_data):
        '''
        checking if the bounding box is collision free
        '''
        left    = int(bb[0])
        top     = int(bb[1])
        right   = int(bb[2])
        bottom  = int(bb[3])
        width = right - left
        height = bottom - top
        if width < self.sfc_minimum_width or height < self.sfc_minimum_height:
            return False
        depth_mat = self.cv_bridge.imgmsg_to_cv2(img_data, desired_encoding='passthrough') * self.depth_scale
        SFC = depth_mat[top:bottom,left:right]
        if SFC.min() <= self.checking_depth:
            return False
        return True

    def planning(self, detection, img_data):
        # Initialising best cost value
        best_cost = -1
        # get goal location in camera frame
        goal_vector = self.get_goal_in_camera_frame()
        feasible = False

        if len(detection) == 0:
            return self.publish_endpoint(feasible, self.best_endpoint, img_data)

        for *xyxy, conf, cls in reversed(detection):
            # verifying safe flight corridors, if one is not collision-free, reject it.
            if not self.verify_SFC(xyxy, img_data):
                continue
            # The projection of the goal is the intersection of the target vector and the image plane.
            feasible = True
            projection = self.project_point_to_pixel(goal_vector)
            if self.pixel_in_bounding_box(projection, xyxy):
                # deprojecting the projection a point in camera frame
                self.best_endpoint = self.deproject_pixel_to_point(projection, self.planning_depth)
                best_cost = np.dot(self.best_endpoint / np.linalg.norm(self.best_endpoint), goal_vector / np.linalg.norm(goal_vector))
                # only visualise the SFC containing the best direction trajectory
                self.visualise_SFC(xyxy)
            else:
                # finding the pixel lying in bounding box that is closest to the goal projection
                closest_pixel = self.closest_pixel_in_bounding_box(projection, xyxy)
                # deprojecting the closest pixel to a point in camera frame
                endpoint = self.deproject_pixel_to_point(closest_pixel, self.planning_depth)
                cost = np.dot(endpoint / np.linalg.norm(endpoint), goal_vector / np.linalg.norm(goal_vector))
                if cost > best_cost:
                    # only visualise the SFC containing the best direction trajectory
                    self.visualise_SFC(xyxy)
                    self.best_endpoint = endpoint
                    best_cost = cost

        return self.publish_endpoint(feasible, self.best_endpoint, img_data)

    def img_callback(self, img_data):
        # Read image
        t0 = time.time()
        self.count += 1
        im0 = self.cv_bridge.imgmsg_to_cv2(img_data, desired_encoding='8UC1')   # convert to uint8 image
        im0s = np.stack((im0,)*3, axis=-1)  # convert to 3 channel image
        assert im0s is not None, 'Image Not Found'

        # Padded resize
        img = letterbox(im0s, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # return path, img, im0s, self.cap
        # for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            # if webcam:  # batch_size >= 1
            #     p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            # else:
            #     p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            s, im0, frame = '', im0s, self.count

            p = Path(str(self.count) + '.jpg')  # to Path
            save_path = str(self.save_dir / p.name)  # img.jpg
            txt_path = str(self.save_dir / 'labels' / p.stem) + f'_{frame}'  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if self.save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if self.save_img or self.view_img:  # Add bbox to image
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)

            # Planning
            self.planning(det, img_data)
            # print(f'Detection & planning done. ({time.time() - t0:.3f}s)')
        # Stream results
        if self.view_img:
            cv2.imshow('Free space detection', im0)
            cv2.waitKey(1)  # 1 millisecond

    # Save results (image with detections)
        # if self.save_img:
            # cv2.imwrite(save_path, im0)
            # print(f" The image with the result is saved in: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='/kingfisher/dodgeros_pilot/unity/depth', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    with open("./planning_config.yaml") as f:
        config = yaml.safe_load(f)
    with torch.no_grad():
        FSD_node = FSDNode(config)
        rospy.spin()
