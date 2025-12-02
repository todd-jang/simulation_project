import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # 설정 파일 경로
    config = os.path.join(
        get_package_share_directory('panda_ai_control'),
        'config',
        'ai_params.yaml' # <-- 수정된 YAML 파일 사용
    )

    return LaunchDescription([
        
        # --- Perception(인지) 계층 ---
        Node(
            package='panda_ai_control',
            executable='bev_vision_node.py',
            name='bev_vision_node',
            parameters=[config],
            output='screen'
        ),
        Node(
            package='panda_ai_control',
            executable='pose_estimation_node.py',
            name='pose_estimation_node',
            parameters=[config],
            output='screen'
        ),
        
        # --- Understanding(이해/판단) 계층 ---
        # [!!!] 새로운 API 엔트리포인트 / LLM 두뇌 노드
        Node(
            package='panda_ai_control',
            executable='task_orchestrator_node.py',
            name='task_orchestrator_node',
            parameters=[config],
            output='screen'
        ),
        
        # --- Action(행동) 계층 ---
        Node(
            package='panda_ai_control',
            executable='dqn_control_node.py', # (목표 실행기로 수정된 버전)
            name='dqn_control_node',
            parameters=[config],
            output='screen'
        ),
        Node(
            package='panda_ai_control',
            executable='main_control_node', # (C++ 하드웨어 인터페이스)
            name='main_control_node',
            parameters=[config], # C++ 노드는 YAML을 직접 읽진 않지만, 일관성을 위해 추가
            output='screen'
        )
    ])
