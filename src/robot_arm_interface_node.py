# robot_arm_interface_node.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
# ⚠️ 주의: 아래 두산 로봇 SDK 관련 임포트는 가상의 코드이며, 실제 SDK에 맞게 수정해야 합니다.
# from doosan_robot_sdk import DRLController 

class RobotArmInterfaceNode(Node):
    def __init__(self):
        super().__init__('robot_arm_interface_node')

        # --- 1. 로봇 컨트롤러 연결 (가상) ---
        # self.robot_controller = DRLController('192.168.1.100')
        # self.robot_controller.connect()
        self.get_logger().info("로봇 컨트롤러 연결 대기 중...")
        
        # --- 2. 구독 설정 ---
        # 추론 노드에서 발행한 행동 명령 구독
        self.action_sub = self.create_subscription(
            Float64MultiArray, '/policy_actions', self.action_callback, 10
        )
        self.get_logger().info("정책 행동 명령 구독 시작. 대기 중...")

    def action_callback(self, msg):
        # 1. 명령 데이터 수신
        joint_velocities = msg.data # [v1, v2, v3, v4, v5, v6] (rad/s)

        # 2. 안전 체크 (선택 사항)
        if len(joint_velocities) != 6: # 관절 개수 확인
             self.get_logger().warn(f"잘못된 관절 수: {len(joint_velocities)}")
             return

        # 3. 로봇 SDK를 통해 명령 전송
        try:
            # ⚠️ 두산 로봇의 '속도 제어 모드' API 호출 (이 부분은 사용자 환경에 맞게 구현)
            # self.robot_controller.set_joint_velocity(joint_velocities) 
            self.get_logger().info(f"로봇에 속도 명령 전송: {joint_velocities[0]:.2f}, ...")

        except Exception as e:
            self.get_logger().error(f"로봇 제어 오류: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RobotArmInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
