# student_policy_inference_node.py
import rclpy
from rclpy.node import Node
import torch
import numpy as np
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge

# ⚠️ 주의: 실제 StudentPolicy 모델 클래스와 가중치 파일 경로는 사용 환경에 맞게 수정해야 합니다.
# 예시로 사용할 더미 클래스 (실제는 StudentPolicy 클래스를 임포트)
class DummyStudentPolicy(torch.nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        # 더미 모델: 입력 크기를 보고 출력(Action)을 더미로 반환
        self.action_dim = action_dim
    def forward(self, depth_tensor, joint_tensor):
        # 실제 모델은 복잡한 CNN+MLP 연산을 수행합니다.
        # 여기서는 더미로 6차원 행동을 반환 (예: 6개의 관절 속도)
        return torch.ones(depth_tensor.shape[0], self.action_dim, device=depth_tensor.device) * 0.1 


class StudentPolicyInferenceNode(Node):
    def __init__(self):
        super().__init__('student_policy_inference_node')
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 1. 모델 로드 및 설정 ---
        self.action_dim = 6  # 두산 로봇 관절 수에 맞게 설정 (예: 6자유도)
        
        # 실제 환경에서는 아래 코드를 사용하며, TensorRT 등으로 변환된 모델 사용 권장
        # self.policy = torch.load("student_policy.pt").to(self.device).eval() 
        self.policy = DummyStudentPolicy(self.action_dim).to(self.device).eval() 

        # --- 2. 구독 설정 ---
        # 카메라 (Depth 이미지) 구독
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10
        )
        # 로봇 관절 상태 구독
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )
        
        # --- 3. 발행 설정 ---
        # 로봇 행동 명령 발행 (robot_arm_interface_node가 구독)
        self.action_pub = self.create_publisher(
            Float64MultiArray, '/policy_actions', 10
        )

        # --- 4. 데이터 저장 변수 ---
        self.latest_depth_tensor = None
        self.latest_joint_tensor = None
        self.MAX_SPEED = 0.5  # 최대 속도 (rad/s) 설정: 안전을 위해 낮게 시작

        # --- 5. 제어 루프 (High Frequency) ---
        # Low Latency를 위해 50Hz (20ms) 또는 100Hz (10ms)로 설정 권장
        self.timer = self.create_timer(0.02, self.control_loop) # 50 Hz

    def depth_callback(self, msg):
        # Image -> CV2 -> Tensor 변환 및 전처리
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # 1. 전처리 (Resize, Noise Filtering, Normalization) - 학습 시와 동일하게!
            preprocessed_tensor = self.preprocess_depth(cv_image)
            self.latest_depth_tensor = preprocessed_tensor
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

    def joint_callback(self, msg):
        # 관절 상태 (위치, 속도 등) Tensor 변환
        # 학습 시 사용한 관측값 순서와 타입에 맞춰야 합니다.
        joint_data = np.array(msg.position, dtype=np.float32)
        self.latest_joint_tensor = torch.tensor(joint_data, device=self.device).unsqueeze(0)

    def preprocess_depth(self, img_array):
        # --- 학습 시 전처리 로직 구현 ---
        # 1. Depth 값 클리핑 및 노이즈 필터링
        img_array = np.clip(img_array, 0, 5000) # 5m 이내만 사용
        # 2. 정규화 (0~1)
        img_normalized = img_array.astype(np.float32) / 5000.0
        # 3. 리사이징 (학습 때 사용한 크기)
        img_resized = cv2.resize(img_normalized, (128, 128))
        # 4. Tensor 변환 (Batch, Channel, H, W)
        tensor_output = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).to(self.device)
        return tensor_output

    def control_loop(self):
        # 데이터가 모두 준비되지 않았으면 대기
        if self.latest_depth_tensor is None or self.latest_joint_tensor is None:
            return

        # 1. 모델 추론 (Inference)
        with torch.no_grad():
            # 두 가지 입력을 모델의 forward 함수에 전달
            raw_action = self.policy(self.latest_depth_tensor, self.latest_joint_tensor)

        # 2. Action Scaling 및 안전장치
        # -1.0 ~ 1.0 범위의 출력을 실제 속도 명령으로 변환
        action_np = raw_action.squeeze(0).cpu().numpy()
        
        # ⚠️ 속도 제한 적용 (안전 장치)
        final_action = np.clip(action_np, -self.MAX_SPEED, self.MAX_SPEED)

        # 3. 명령 메시지 발행
        action_msg = Float64MultiArray()
        action_msg.data = final_action.tolist()
        self.action_pub.publish(action_msg)

        # 사용한 데이터는 다음 스텝에서 재사용될 수 있도록 유지 (DAgger 스타일)
        # 만약 메모리 누수가 발생한다면 여기서 self.latest_depth_tensor = None 등을 추가해야 함

def main(args=None):
    rclpy.init(args=args)
    node = StudentPolicyInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
