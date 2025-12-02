#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
import message_filters
import cv2
import numpy as np
from cv_bridge import CvBridge
import tf2_ros

class BevVisionNode(Node):
    def __init__(self):
        super().__init__('bev_vision_node')
        self.get_logger().info('BEV 비전 노드 (Python) 초기화 중...')

        # 파라미터 선언
        self.declare_parameter('wrist_cam_topic', '/wrist_camera/image_raw')
        self.declare_parameter('surround_cam_topic', '/surround_camera/image_raw')
        self.declare_parameter('bev_image_topic', '/bev_image')
        
        # 파라미터 값 가져오기
        wrist_cam_topic = self.get_parameter('wrist_cam_topic').value
        surround_cam_topic = self.get_parameter('surround_cam_topic').value
        bev_image_topic = self.get_parameter('bev_image_topic').value

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 다중 카메라 입력을 시간 기준으로 동기화
        self.wrist_sub = message_filters.Subscriber(self, Image, wrist_cam_topic)
        self.surround_sub = message_filters.Subscriber(self, Image, surround_cam_topic)

        # ApproximateTimeSynchronizer는 두 토픽이 정확히 동시에 오지 않아도 처리
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.wrist_sub, self.surround_sub], 10, 0.1)
        self.ts.registerCallback(self.image_callback)

        # BEV 이미지 발행
        self.bev_pub = self.create_publisher(Image, bev_image_topic, 10)
        
        self.get_logger().info('BEV 노드 준비 완료. 카메라 영상 대기 중...')

    def image_callback(self, wrist_msg, surround_msg):
        try:
            # ROS 이미지를 OpenCV 이미지로 변환
            wrist_cv = self.bridge.imgmsg_to_cv2(wrist_msg, 'bgr8')
            surround_cv = self.bridge.imgmsg_to_cv2(surround_msg, 'bgr8')

            # 카메라 파라미터와 TF(Transform) 정보 가져오기 (실제 구현 필요)
            # wrist_cam_info = ...
            # surround_cam_info = ...
            # wrist_to_base_tf = self.tf_buffer.lookup_transform('panda_link0', wrist_msg.header.frame_id, rclpy.time.Time())

            # --- 핵심 BEV 생성 로직 (Placeholder) ---
            # 여기에 실제 BEV 생성 알고리즘 (IPM, 트랜스포머 기반 등)이 들어갑니다.
            # 지금은 두 이미지를 단순히 합치는 것으로 시뮬레이션합니다.
            
            h1, w1 = wrist_cv.shape[:2]
            h2, w2 = surround_cv.shape[:2]
            
            # 크기 조절 (예시)
            surround_cv_resized = cv2.resize(surround_cv, (w1, h1))
            
            # BEV 이미지 시뮬레이션 (상단: 주변, 하단: 손목)
            bev_image = np.vstack([surround_cv_resized, wrist_cv])
            # ------------------------------------

            # BEV 이미지를 ROS 메시지로 변환하여 발행
            bev_msg = self.bridge.cv2_to_imgmsg(bev_image, 'bgr8')
            bev_msg.header.stamp = self.get_clock().now().to_msg()
            bev_msg.header.frame_id = "panda_link0" # 기준 좌표계
            self.bev_pub.publish(bev_msg)

        except Exception as e:
            self.get_logger().error(f'BEV 생성 중 오류: {e}')

def main(args=None):
    rclpy.init(args=args)
    bev_node = BevVisionNode()
    rclpy.spin(bev_node)
    bev_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
