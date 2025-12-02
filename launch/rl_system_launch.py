from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 패키지 공유 디렉토리 경로 설정
    pkg_name = 'doosan_rl_sim2real' # 사용자의 ROS 2 패키지명으로 변경
    pkg_share_dir = get_package_share_directory(pkg_name)
    
    # 모델 파일 경로 설정 (예시: 패키지 내 'models' 폴더에 student_policy.pt 파일이 있다고 가정)
    model_path = os.path.join(pkg_share_dir, 'models', 'student_policy.pt')
    
    return LaunchDescription([
        
        ## 1. Camera Driver Node (D455 센서 데이터 획득)
        Node(
            package=pkg_name,
            executable='camera_driver_node',
            name='realsense_driver',
            output='screen',
            parameters=[
                {'publish_frequency': 30.0} # D455 FPS 설정
            ]
        ),

        ## 2. Student Policy Inference Node (추론 모델 실행)
        Node(
            package=pkg_name,
            executable='student_policy_inference_node',
            name='policy_inference_brain',
            output='screen',
            parameters=[
                {'model_path': model_path},          # 학습된 모델 파일 경로
                {'max_speed_limit': 0.1},            # ⚠️ 안전을 위해 초기 속도 제한 설정 (rad/s)
                {'inference_rate': 50.0},             # 제어 주기 (50Hz = 20ms 지연)
                {'device': 'cuda'}                   # GPU 사용 설정
            ],
            # 토픽 통신 확인: 
            # - Subs: /camera/depth/image_raw, /joint_states
            # - Pubs: /policy_actions
        ),

        ## 3. Robot Arm Interface Node (로봇 하드웨어 명령 전달)
        Node(
            package=pkg_name,
            executable='robot_arm_interface_node',
            name='doosan_controller_bridge',
            output='screen',
            # 토픽 통신 확인: 
            # - Subs: /policy_actions
        ),
        
        # ⚠️ (선택 사항) 로봇 하드웨어 드라이버 노드도 필요하다면 여기에 추가
        # Node(
        #     package='doosan_ros2_driver', 
        #     executable='doosan_robot_node', 
        #     name='robot_hw_driver'
        # ),
    ])
