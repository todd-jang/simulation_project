# camera_driver_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs # Intel RealSense SDK
import numpy as np

class CameraDriverNode(Node):
    def __init__(self):
        super().__init__('camera_driver_node')
        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.get_logger().info('RealSense Camera Driver 초기화 중...')

        # --- 1. RealSense 파이프라인 설정 ---
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Depth 스트림 활성화 (예: 640x480, 30 FPS)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # ⚠️ 필터 설정 (Sim-to-Real Gap 해소)
        # 현실의 노이즈를 시뮬레이션 데이터처럼 줄여주는 필터 사용
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling = rs.hole_filling_filter() # 결측치 보정

        self.pipeline.start(self.config)
        self.get_logger().info("RealSense 파이프라인 시작 완료.")

        # --- 2. 타이머 설정 (카메라 FPS에 맞춰야 함) ---
        # 30 FPS 카메라에 맞게 약 33ms 주기 설정
        self.timer = self.create_timer(1.0/30.0, self.timer_callback) 

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return

        # 3. RealSense 필터 적용
        filtered_frame = self.spatial_filter.process(depth_frame)
        filtered_frame = self.temporal_filter.process(filtered_frame)
        filtered_frame = self.hole_filling.process(filtered_frame)
        
        # 4. 데이터 추출 및 NumPy 변환
        depth_image = np.asanyarray(filtered_frame.get_data())

        # 5. ROS Image 메시지 변환 및 발행
        try:
            # 16-bit Depth 이미지 (Z16) 그대로 발행
            ros_image = self.bridge.cv2_to_imgmsg(depth_image, encoding='16UC1')
            ros_image.header.stamp = self.get_clock().now().to_msg()
            self.publisher_.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"Image 발행 오류: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # 파이프라인 정지
    node.pipeline.stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
