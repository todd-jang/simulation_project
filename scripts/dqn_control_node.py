#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import message_filters

class DqnControlNode(Node):
    def __init__(self):
        super().__init__('dqn_control_node')
        self.get_logger().info('DQN 제어 노드 (Python) 초기화 중...')

        self.declare_parameter('pose_topic', '/detected_object_pose')
        self.declare_parameter('joint_state_topic', '/joint_states') # 실제 Franka 토픽
        self.declare_parameter('ai_command_topic', '/panda_ai_commands')
        
        pose_topic = self.get_parameter('pose_topic').value
        joint_state_topic = self.get_parameter('joint_state_topic').value
        ai_command_topic = self.get_parameter('ai_command_topic').value

        # --- DQN 모델 로드 (Placeholder) ---
        # self.dqn_model = load_model('path/to/dqn.pth')
        # self.get_logger().info('DQN 모델 로드 완료 (시뮬레이션).')
        # ---------------------------------

        # 상태(State) 입력을 위한 구독자 (자세, 관절)
        self.pose_sub = message_filters.Subscriber(self, PoseStamped, pose_topic)
        self.joint_sub = message_filters.Subscriber(self, JointState, joint_state_topic)
        
        # 두 상태를 동기화하여 콜백 실행
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.pose_sub, self.joint_sub], 10, 0.5)
        self.ts.registerCallback(self.state_callback)

        # 행동(Action) 발행 (main_control_node로 전송)
        self.action_pub = self.create_publisher(Float64MultiArray, ai_command_topic, 10)
        
        self.get_logger().info('DQN 노드 준비 완료. 상태 정보 대기 중...')

    def state_callback(self, pose_msg, joint_msg):
        try:
            # --- 1. 상태(State) 정의 ---
            # DQN 모델의 입력으로 사용할 상태 벡터(State Vector)를 구성합니다.
            
            # 물체 위치
            obj_pos = np.array([
                pose_msg.pose.position.x,
                pose_msg.pose.position.y,
                pose_msg.pose.position.z
            ])
            
            # 현재 로봇 관절 위치 (Franka는 7개 관절)
            # joint_msg.name 순서가 다를 수 있으므로 실제로는 정렬 필요
            current_joints = np.array(joint_msg.position[:7])
            
            # 상태 벡터 (물체 위치 + 현재 관절 위치)
            state_vector = np.concatenate([obj_pos, current_joints])
            
            # --- 2. 행동(Action) 계산 (Placeholder) ---
            # self.dqn_model.eval()
            # action_tensor = self.dqn_model(torch.tensor(state_vector).float())
            # action = action_tensor.detach().numpy()
            
            # (시뮬레이션: 현재 관절 위치를 그대로 유지하는 명령 전송)
            action = current_joints 
            # (실제 예시: action = current_joints + delta_joints)
            # ----------------------------------------
            
            # Float64MultiArray 메시지로 변환
            action_msg = Float64MultiArray()
            action_msg.data = action.tolist()
            
            # 행동 명령 발행
            self.action_pub.publish(action_msg)

        except Exception as e:
            self.get_logger().error(f'DQN 추론 중 오류: {e}')

def main(args=None):
    rclpy.init(args=args)
    dqn_node = DqnControlNode()
    rclpy.spin(dqn_node)
    dqn_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
