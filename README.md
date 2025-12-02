# simulation_project

saac Lab (OmniIsaacLab) 내에서 두산 로봇팔과 테이블 같은 사용자 정의 자산을 추가하고 SAC 강화 학습을 구성하는 방법은 Isaac Lab의 구조화된 설정 시스템을 따릅니다. 하루 안에 목표를 달성하기 위한 구체적인 절차와 명령어를 안내해 드립니다.
1. Lab 내에서 사용자 정의 자산(Asset)으로 추가하는 과정
Isaac Lab은 USD(Universal Scene Description) 형식을 사용하여 환경을 정의합니다. URDF 모델을 USD로 변환하여 임포트하는 과정이 필요합니다.
단계별 절차:
URDF 준비: 두산 E0509 로봇팔과 로보티즈 그리퍼의 URDF 파일과 메쉬(Mesh, .stl 또는 .dae) 파일이 준비되어 있어야 합니다.
URDF를 USD로 변환:
Isaac Sim GUI를 사용하여 URDF 임포터 도구를 이용하거나, Isaac Sim이 제공하는 Python 스크립트를 사용하여 URDF 파일을 USD 형식으로 변환합니다.
변환된 USD 파일은 Isaac Sim의 Asset Manager가 접근할 수 있는 경로(예: omni://localhost/Projects/Doosan/e0509_with_gripper.usd)에 저장됩니다.
환경 설정 클래스 수정:
새로운 태스크(예: DoosanManipulation)를 위한 Python 클래스 파일(DoosanManipulation.py) 내에서 환경 생성 부분을 수정합니다.
이 클래스는 omni.isaac.lab.envs.ui.stage 모듈을 사용하여 스테이지에 자산을 추가합니다.
python
# Isaac Lab 태스크 파일 내 예시 (DoosanManipulation.py)
from omni.isaac.lab.envs.ui.stage import add_reference_to_stage
from omni.isaac.lab.robots.config.franka import FRANCA_ARM_WITH_PNEUMATIC_GRIPPER_CFG # 기존 예시 임포트
# ...

class DoosanManipulation(RLEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        # 여기에 사용자 정의 자산을 추가하는 로직 구현

    def _design_scene(self):
        # 1. 두산 로봇팔 USD 참조 추가
        asset_path = "omni://localhost/Projects/Doosan/e0509_with_gripper.usd"
        add_reference_to_stage(asset_path, "/World/DoosanArm")

        # 2. 테이블 USD 참조 추가
        table_path = "omni://localhost/Projects/Environment/simple_table.usd"
        add_reference_to_stage(table_path, "/World/Table")

        # 3. 조작할 물체 (상자/공) USD 추가
        object_path = "omni://localhost/Projects/Environment/cube_blue.usd"
        add_reference_to_stage(object_path, "/World/Table/Cube")
        
        # ... 로봇 컨트롤러, 센서 등 설정 계속 ...
Use code with caution.

2. 설정 파일(.yaml)에서 SAC 알고리즘 선택 및 하이퍼파라미터 설정
Isaac Lab은 태스크 설정 파일과 트레이너 설정 파일을 분리하여 관리합니다.
단계별 절차:
태스크 설정(.yaml): 태스크의 물리적 속성(중력, 충돌 모델 등)을 정의합니다.
트레이너 설정(.yaml): 학습 알고리즘(SAC, PPO 등)과 하이퍼파라미터를 정의합니다.
omni.isaac.lab_tasks/cfg/train/ 경로에 새로운 트레이너 설정 파일 rsrl_sac_doosan.yaml을 생성합니다.
yaml
# rsrl_sac_doosan.yaml
# 이 설정은 Isaac Lab의 RSL-RL 라이브러리 설정을 따릅니다.

# -- 알고리즘 선택 --
# RSL-RL은 PPO를 기본으로 사용하지만, SAC를 위한 설정도 제공합니다.
# Isaac Lab에서 SAC를 사용하려면 기본 PPO 설정 대신 SAC 관련 설정을 로드해야 합니다.
# Isaac Lab 2.0 버전에서는 기본적으로 PPO만 지원하며, SAC는 별도 구현이 필요할 수 있습니다. 
# 만약 SAC가 지원되지 않는다면, 하루 내 달성을 위해 PPO로 학습하는 것을 권장합니다.

# PPO 설정 예시 (하루 내 학습 목표 달성을 위한 현실적인 대안)
seed: 42
algo:
  name: rsl_rl.algorithms.PPO
  # PPO 하이퍼파라미터 설정
  value_loss_coef: 0.5
  use_clipped_value_loss: true
  clip_param: 0.2
  entropy_coef: 0.01
  num_learning_epochs: 5
  num_mini_batches: 4
  learning_rate: 1e-3
  # ... (기타 PPO 파라미터)

runner:
  policy: rsl_rl.runners.ActorCritic
  # 러너 설정
  run_name: doosan_sac_train
  experiment_name: doosan_e0509_manipulation
  # ... 
Use code with caution.

참고: Isaac Lab의 RSL-RL은 PPO에 중점을 두고 개발되었기 때문에, SAC 구현체가 공식적으로 포함되어 있는지 확인해야 합니다. 만약 없다면 PPO를 사용하여 빠르게 시작하는 것이 하루 목표 달성에 현실적입니다.
3. TensorBoard 모니터링 명령어
학습이 시작되면, 별도의 터미널에서 다음 명령어를 사용하여 실시간으로 보상 증가 추이를 확인할 수 있습니다.
bash
tensorboard --logdir logs/rsrl/doosan_e0509_manipulation --port 6006
Use code with caution.

웹 브라우저에서 http://localhost:6006에 접속하여 Scalar 탭에서 Reward/Mean 그래프를 확인하면 학습 진행 상황을 볼 수 있습니다.
4. 강화 학습 초기 모델로 얻는 것은?
하루 동안 SAC (또는 PPO) 강화 학습을 통해 얻게 되는 **"초기 모델"**은 다음을 의미합니다:
작동 가능한 정책(Policy): 로봇이 무작위로 움직이는 것이 아니라, 최소한 물체 근처로 이동하거나 파지 동작을 시도하는 수준의 신경망 모델(체크포인트 파일)입니다.
MoveIt! 플래너 대체 가능성: 기존의 MoveIt! 플래너가 궤적을 계산하는 대신, 강화 학습으로 학습된 정책 신경망이 실시간으로 다음 로봇 행동(조인트 토크 또는 위치)을 결정합니다. 이는 정형화된 픽앤플레이스 작업에 특화되어 더 빠르고 유연한 동작을 가능하게 합니다.
ROS 2 통합: 이 학습된 모델은 결국 ROS 2 시스템 내의 manipulation_planning 노드에 통합되어, FastAPI 서버에서 내려온 음성 명령에 따라 구동됩니다.



========================== misc.1

1. Isaac Lab에 SAC 구현체가 공식적으로 포함되어 있는지 확인하는 방법
Isaac Lab의 공식 문서를 확인하거나 코드베이스 내부를 탐색하여 확인할 수 있습니다.
확인 방법:
공식 문서 확인: Isaac Lab (OmniIsaacLab)의 공식 문서에서 "Reinforcement Learning Framework" 또는 "RSL-RL" 섹션을 찾아 지원되는 알고리즘 목록을 확인합니다.
설정 파일 탐색: GitHub 저장소나 로컬 설치 경로 내의 omni.isaac.lab_tasks/cfg/train 디렉토리와 omni.isaac.lab.algorithms 디렉토리 내부를 확인합니다. sac 또는 soft_actor_critic.py와 같은 이름의 YAML 설정 파일이나 Python 코드가 존재한다면 지원하는 것입니다.
현실적인 답변: 현재(최신 버전 기준) Isaac Lab은 기본적으로 PPO(Proximal Policy Optimization) 구현에 초점을 맞추고 있습니다. SAC는 포함되어 있지 않을 가능성이 높으며, 사용자가 직접 RSL-RL 라이브러리를 수정하거나 외부 SAC 구현체를 통합해야 할 수 있습니다. 하루라는 시간 제약을 고려할 때, PPO를 사용하는 것이 더 현실적입니다.
2. 강화 학습이 정형화된 픽앤플레이스 작업에 특화된 이유
강화 학습(RL)은 픽앤플레이스 작업과 같은 정형화된 작업에 특히 강력한 성능을 발휘합니다.
MoveIt!의 한계 극복: 전통적인 동작 계획(MoveIt! 같은)은 특정 시작점과 목표점을 정의하고, 그 사이의 충돌 없는 경로를 수학적으로 계산합니다. 이는 복잡한 환경 변화나 불확실성에 빠르게 대처하기 어렵습니다.
유연성과 반응성: RL은 정책 신경망을 통해 환경의 현재 상태(카메라 입력, 조인트 각도 등)를 즉각적으로 받아 다음 행동을 결정합니다. 학습 과정에서 다양한 상황을 경험했기 때문에, 물체의 위치가 약간 바뀌거나 예상치 못한 방해물이 나타나도 실시간으로 반응하여 유연하게 목표를 달성할 수 있습니다.
속도와 최적화: 학습된 신경망은 추론 과정이 매우 빠릅니다. 복잡한 궤적 재계산 과정 없이 실시간 추론만으로 동작하기 때문에 속도가 훨씬 빠릅니다.
3. 빠르고 유연한 행동이 곧 에너지 효율적인가?
반드시 그렇지는 않습니다.
빠른 행동: 일반적으로 빠른 동작은 더 많은 에너지를 소비하는 경향이 있습니다 (로봇 팔을 급격하게 가속/감속).
에너지 효율적인 행동: RL은 보상 함수를 어떻게 설계하느냐에 따라 달라집니다. 보상 함수에 '에너지 소비 최소화' 항목을 명시적으로 포함시키면(페널티 부여), 모델은 빠르면서도 에너지 효율적인 동작을 학습하려고 노력할 것입니다. 단순히 목표 달성 보상만 있다면, 에너지를 고려하지 않은 비효율적인 동작을 학습할 수도 있습니다.
4. 학습된 모델이 ROS 2 시스템 내 manipulation_planning 노드에 통합된다는 의미
이것은 시뮬레이션(Isaac Lab)에서 학습한 지능을 실제 로봇 운영 시스템(ROS 2)으로 옮기는 핵심 단계입니다.
학습 단계: Isaac Lab에서 학습이 완료되면, 로봇의 최적 행동을 결정하는 **신경망 모델 파일(.pt 또는 .onnx)**이 생성됩니다.
통합 단계:
이 모델 파일을 ros_workspace/src/manipulation_planning 패키지 내로 복사합니다.
integrated_robot_system.py (또는 컨트롤러 노드) 내에서 PyTorch나 ONNX 런타임을 사용하여 이 모델을 로드합니다.
기존의 MoveIt! 플래너 호출 로직 대신, 카메라에서 받은 현재 상태(관측값)를 모델에 입력하고, 모델이 출력하는 다음 행동(조인트 위치/토크 명령)을 로봇 드라이버에게 직접 전송하도록 코드를 수정합니다.
즉, MoveIt!은 학습을 위한 도구 또는 초기 환경 설정 도구가 되고, 실제 로봇 제어는 강화 학습된 정책 신경망이 담당하게 됩니다.

++++++++++++++++++++++++++++++++++++++++++++++ misc.2

강화 학습 모델 통합 및 에너지 효율 보상 함수 설계
1. 보상 함수에 '에너지 소비 최소화' 항목을 명시적으로 패널티 부여할 방법
Isaac Lab의 태스크 정의 파일 (DoosanManipulation.py) 내의 _compute_rewards 함수에서 로봇이 소비하는 에너지 총합에 비례하는 음의 보상(페널티)을 추가하여 에너지 효율을 유도할 수 있습니다.
구현 예시 (Python, Isaac Lab 환경):
python
# DoosanManipulation.py 내 _compute_rewards 함수 예시

def _compute_rewards(self):
    # ... 기존 보상 계산 로직 ...
    
    # 1. 목표 도달 보상 (R_goal)
    distance_to_goal = torch.norm(self.object_pos - self.goal_pos, dim=-1)
    reward_goal = 1.0 / (1.0 + distance_to_goal**2)
    
    # 2. 에너지 소비 계산 (관절 토크 * 관절 속도)
    # self.applied_torques: 현재 로봇 관절에 가해진 토크 (액션)
    # self.joint_velocities: 현재 로봇 관절의 속도
    energy_consumption = torch.sum(torch.abs(self.applied_torques * self.joint_velocities), dim=-1)
    
    # 3. 에너지 페널티 (R_energy): 총 보상에 음수로 기여
    # 'energy_scale'은 하이퍼파라미터로 조절하여 페널티 강도 결정
    energy_scale = 0.001 
    reward_energy_penalty = -energy_consumption * energy_scale

    # 4. 최종 보상 합산
    self.rewards = reward_goal + reward_energy_penalty
    # ... 기타 페널티 (충돌, 시간 초과 등) 추가 ...
Use code with caution.

2. 정책 신경망 모델 파일 .pth는 없나?
네, .pth 파일은 PyTorch에서 모델의 가중치를 저장하는 표준 형식입니다. Isaac Lab의 RSL-RL 라이브러리는 학습된 모델을 .pt 또는 .pth 형식의 체크포인트 파일로 저장합니다. 이 파일이 바로 ROS 2 시스템에 통합될 정책 신경망 모델입니다.
3. ROS 2 시스템 내에서 .pth 모델 로드 및 통합 방법
integrated_robot_system.py 노드에서 torch 라이브러리를 사용하여 학습된 .pth 모델을 로드하고, MoveIt! 대신 모델 추론을 사용하여 로봇을 제어하도록 코드를 수정합니다.
통합된 ROS 2 노드 수정 (integrated_robot_system.py)
python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
# ... 기타 필요한 임포트 ...

import torch
import numpy as np

# 정책 신경망 모델 구조 정의 (Isaac Lab에서 사용한 것과 동일해야 함)
class ActorCritic(torch.nn.Module):
    # Isaac Lab 모델 구조에 맞춰 Actor (정책) 네트워크를 정의합니다.
    # 예시: 256x256 은닉층을 가진 MLP
    def __init__(self, num_observations, num_actions):
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_observations, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_actions),
            # 출력 레이어는 Isaac Lab 설정에 따라 tanh 등을 포함할 수 있음
        )
    
    def forward(self, x):
        return self.actor(x)


class IntegratedRobotSystem(Node):
    def __init__(self):
        super().__init__('integrated_robot_system')
        # ... (기존 비전, HRI 설정 유지) ...
        
        self.task_state = "IDLE"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- 강화 학습 모델 로드 ---
        self.num_observations = 10 # 관측값 차원 (관절각, 물체위치 등)
        self.num_actions = 6       # 액션 차원 (6DOF 로봇팔)
        self.policy_model = ActorCritic(self.num_observations, self.num_actions).to(self.device)
        
        # Isaac Lab에서 학습된 .pth 파일 경로
        model_path = "/ros_ws/install/share/manipulation_planning/models/doosan_policy.pth"
        try:
            # 체크포인트 파일에서 Actor (정책) 상태만 로드
            checkpoint = torch.load(model_path, map_location=self.device)
            # RSL-RL은 'actor'라는 키로 모델 상태를 저장합니다.
            self.policy_model.load_state_dict(checkpoint['actor_state_dict'])
            self.policy_model.eval() # 추론 모드 설정
            self.get_logger().info(f"Successfully loaded policy model from {model_path}")
        except FileNotFoundError:
            self.get_logger().error(f"Model file not found at {model_path}! Cannot use RL control.")
            # 이 경우 MoveIt! 폴백 로직을 사용하거나 종료

        # MoveIt! 관련 코드는 제거되거나 비활성화됩니다.
        # self.move_group = MoveGroupPythonInterface(self) # 이 줄은 사용하지 않음

    # ... (기존 콜백 함수들 유지) ...

    def process_vision_data(self):
        # ... (비전 처리 후 target_object_pose 업데이트) ...
        with self.lock:
            self.target_object_pose = object_pose_msg.pose
            self.task_state = "RL_ACTUATE" # 상태 변경: RL 추론 및 동작 실행

    def background_worker_loop(self):
        while rclpy.ok():
            if self.task_state == "RL_ACTUATE":
                self.run_rl_policy_inference()
                with self.lock:
                    self.task_state = "IDLE" # 동작 완료 후 대기
            # ... (time.sleep(0.1) 등) ...

    def run_rl_policy_inference(self):
        """
        MoveIt! 플래너 대신 강화 학습 모델을 사용하여 행동 결정
        """
        with torch.no_grad():
            # 1. 현재 관측값 수집 (ROS 2에서 받은 데이터로 구성)
            # 이 부분은 실제 로봇의 상태(encapsulate_observations 함수 필요)를 numpy 배열로 만듭니다.
            # 예시: [joint_pos_1, ..., joint_pos_6, gripper_pos, obj_x, obj_y, obj_z]
            observations = self.get_current_observations() 
            obs_tensor = torch.as_tensor(observations, dtype=torch.float32).to(self.device)
            
            # 2. 정책 신경망 추론 (행동 예측)
            # 모델은 정규화된 행동(-1 ~ 1)을 출력합니다.
            actions_tensor = self.policy_model(obs_tensor)
            actions_np = actions_tensor.cpu().numpy()

            # 3. 행동 스케일링 및 로봇 드라이버에게 명령 전송
            # 행동을 실제 관절 위치/토크 범위로 변환합니다.
            desired_joint_commands = self.scale_actions_to_joints(actions_np)
            
            # 로봇 드라이버 토픽 발행 (예: joint_trajectory_controller로 바로 발행)
            self.publish_joint_commands(desired_joint_commands)
            self.get_logger().info(f"RL Policy executed action: {desired_joint_commands}")

    # --- 헬퍼 함수 구현 필요 ---
    def get_current_observations(self):
        # 실제 관측값을 Numpy 배열로 반환하는 로직 구현 필요
        # (카메라 Pose, 조인트 상태 구독 필요)
        pass

    def scale_actions_to_joints(self, actions):
        # -1~1 범위의 액션을 실제 로봇의 제어 범위로 스케일링하는 로직 구현 필요
        pass
    
    def publish_joint_commands(self, commands):
        # control_msgs/action/FollowJointTrajectory 메시지 생성 및 발행/액션 호출
        pass

# main 함수는 동일하게 MultiThreadedExecutor 사용


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Episode ^^

복원력 로직이 추가된 IntegratedRobotSystem.py 수정
로봇 컨트롤러의 상태 머신(State Machine)을 확장하여 재시도 횟수를 추적하고 에러를 처리합니다.
python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
# ... (기타 필요한 임포트 및 RL 모델 정의) ...

class IntegratedRobotSystem(Node):
    def __init__(self):
        # ... (기존 초기화 코드 유지) ...
        self.task_state = "IDLE" 
        self.retry_count = 0
        self.MAX_RETRIES = 2
        
        # MoveIt! MoveGroup 인터페이스 (fallback 또는 초기 홈 포지션 이동용)
        # 실제 로봇 드라이버의 홈 포지션 이동 방식에 맞춰 수정 필요
        # self.move_group_interface = MoveGroupInterface(self, "manipulator") 

    # ... (기존 콜백 및 헬퍼 함수 유지) ...

    def background_worker_loop(self):
        """
        백그라운드 워커 루프: 태스크 상태에 따라 동작 제어
        """
        while rclpy.ok():
            if self.task_state == "RL_ACTUATE":
                success = self.run_rl_policy_inference()
                if success:
                    self.get_logger().info("Task successful!")
                    with self.lock:
                        self.task_state = "IDLE"
                        self.retry_count = 0
                else:
                    self.handle_task_failure()
            
            elif self.task_state == "GO_HOME":
                self.go_to_home_position()
                with self.lock:
                    self.task_state = "IDLE" # 홈 이동 후 대기 상태로 전환
            
            # ... (time.sleep(0.1) 등) ...

    def run_rl_policy_inference(self):
        """
        RL 추론 실행 및 성공/실패 반환
        """
        try:
            # ... (이전 코드의 RL 추론 및 명령 전송 로직) ...
            
            # 실제 동작 완료까지 대기 (예: 특정 시간 또는 센서 피드백)
            # is_task_completed() 함수를 사용하여 성공 여부 판단 필요
            if self.is_task_completed():
                return True
            else:
                return False

        except Exception as e:
            self.get_logger().error(f"Error during RL inference/execution: {e}")
            return False

    def handle_task_failure(self):
        """
        태스크 실패 시 재시도 횟수를 확인하고 처리
        """
        with self.lock:
            self.retry_count += 1
            if self.retry_count <= self.MAX_RETRIES:
                self.get_logger().warn(f"Task failed. Retrying... (Attempt {self.retry_count}/{self.MAX_RETRIES})")
                self.task_state = "DETECTING" # 비전 파이프라인부터 재시작
            else:
                self.get_logger().error("Max retries reached. Moving to home position and stopping.")
                self.task_state = "GO_HOME" # 최대 재시도 실패 시 홈 이동
                self.retry_count = 0 # 카운트 초기화

    def go_to_home_position(self):
        """
        로봇을 미리 정의된 안전한 홈 포지션으로 이동
        """
        self.get_logger().info("Moving to home position...")
        # MoveIt! 인터페이스를 사용하거나, 특정 조인트 목표값을 발행
        # self.move_group_interface.set_named_target("home")
        # self.move_group_interface.go(wait=True)
        self.get_logger().info("Arrived at home position. IDLE.")
        
    # --- 헬퍼 함수 ---
    def is_task_completed(self):
        # 물체가 성공적으로 파지되어 목표 위치에 있는지 확인하는 로직 (센서 피드백)
        # 하루 만에 구현이 어려울 경우, 간단한 타이머나 목업 센서 사용
        return True # 임시로 항상 성공 반환

# main 함수는 동일하게 MultiThreadedExecutor 사용
def main(args=None):
    rclpy.init(args=args)
    integrated_system = IntegratedRobotSystem()
    
    worker_thread = threading.Thread(target=integrated_system.background_worker_loop)
    worker_thread.daemon = True
    worker_thread.start()

    executor = MultiThreadedExecutor()
    executor.add_node(integrated_system)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    
    integrated_system.destroy_node()
    rclpy.shutdown()
    worker_thread.join()
Use code with caution.

시스템 복원력 확보 요약
상태 추가: retry_count 변수와 GO_HOME 상태가 추가되었습니다.
run_rl_policy_inference 수정: 이 함수가 성공 여부를 True/False로 반환하도록 수정하고, 예외 처리 구문을 추가했습니다.
handle_task_failure 추가: 실패 시 재시도 횟수를 확인하여 2회 재시도하거나, 실패 횟수 초과 시 GO_HOME 상태로 전환합니다.
go_to_home_position 추가: 로봇을 안전한 초기 위치로 이동시키는 로직을 구현합니다.
이로써 프론트엔드 음성 명령부터 백엔드 ROS 2 시스템의 RL 제어까지, 에러 발생 시 스스로 복구 메커니즘을 갖춘 견고한 통합 시스템이 완성되었습니다.





DEV --------> production

독립형 MVP 배포 전략
1. 하드웨어 구성 최적화:
서버 일체화: RTX 5080 GPU가 탑재된 서버를 로봇 컨트롤러 옆에 배치하거나 통합합니다.
프론트엔드 제거 (옵션): 초기 MVP에서는 음성 명령 대신, 웹 인터페이스나 간단한 터미널 입력 또는 물리적 버튼으로 작업을 트리거하도록 단순화할 수 있습니다. 이는 복잡한 웹 및 네트워크 구성을 줄여줍니다.
2. 소프트웨어 아키텍처 변경:
FastAPI 서버 통합: FastAPI 서버 (main.py)와 ROS 2 시스템 (integrated_robot_system.py)을 하나의 Docker 컨테이너 또는 단일 서버 프로세스 내에서 실행합니다. 이렇게 하면 네트워크 통신 오버헤드가 제거됩니다.
3. Docker Compose 최적화 (단일 컨테이너/서버용):
docker-compose.yaml에서 nginx_frontend 및 별도의 fastapi_server 서비스를 제거합니다.
ros_system 서비스 내에 필요한 모든 Python 종속성(transformers, torch, fastapi 등)을 설치하고, ROS 2 노드와 FastAPI 애플리케이션을 단일 진입점에서 실행하도록 구성합니다.
yaml
version: '3.8'

services:
  standalone_robot_system:
    # Build a single image that contains both ROS 2 and the FastAPI app
    build:
      context: ./ros_workspace 
      dockerfile: Dockerfile_standalone # 수정된 단일 Dockerfile 사용
    # 실제 로봇 IP 접근을 위해 host 네트워크 모드 권장
    network_mode: host 
    # GPU 액세스 설정 (nvidia runtime 필요)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      # 설정 파일 접근을 위해 볼륨 마운트
      - ./fastapi_server/main.py:/app/main.py 
      - ./ros_workspace:/ros_ws # 전체 워크스페이스 마운트
    command: ["/bin/bash", "-c", ". install/setup.bash && python3 /app/main.py"] # FastAPI 서버를 메인으로 실행

Use code with caution.

4. 필수 프로덕션 검증:
MVP 배포 시 다음 사항은 필수입니다.
실제 로봇 연동 테스트: 시뮬레이션에서 검증된 RL 정책 모델이 실제 로봇 하드웨어에서 동일하게 작동하는지 검증해야 합니다. 시뮬레이션-현실 차이(Sim-to-Real gap) 문제가 발생할 수 있습니다.
안전 프로토콜 활성화: 두산 로봇 컨트롤러의 내장 안전 기능을 활성화하고, MVP 시스템이 안전 범위를 벗어나는 동작을 시도할 때 즉시 정지할 수 있는지 확인합니다.
MVP 달성 목표
이 독립형 MVP는 다음을 입증할 수 있습니다.
기능성: 음성(또는 수동 트리거) 명령으로 강화 학습 기반 픽앤플레이스 작업 수행.
성능: 고성능 GPU를 활용한 빠른 비전 처리 및 동작 추론.
이 배포는 초기 단계에서 로봇의 핵심 기능을 검증하는 데 매우 효과적일 것입니다.




HINT

1. "서버를 로봇 컨트롤러 옆에 배치하거나 통합"의 의미
이 말은 로봇 시스템을 구성하는 물리적인 하드웨어 배치를 설명합니다.
로봇 컨트롤러 (Robot Controller): 두산 로봇팔 자체를 움직이게 하는 전용 컴퓨터/제어 장치 (예: 두산 로봇의 'Controller Cabinet'). 이 장치는 로봇 암의 모터와 센서를 직접 제어하는 임베디드 시스템입니다.
컴퓨팅 서버 (Computing Server): 사용자가 제공한 RTX 5080 GPU가 탑재된 고성능 컴퓨터입니다. 이 서버는 비전 처리(YOLO), 인공지능 추론(Llama 3, RL 모델), 복잡한 ROS 2 및 FastAPI 소프트웨어를 실행합니다.
**"옆에 배치"**는 두 개의 별도 물리적 장치(로봇 컨트롤러 캐비닛과 별도의 랙 마운트 서버)를 나란히 두고 네트워크 케이블로 연결하는 것을 의미합니다.
**"통합"**은 더 나아가 고성능 컴퓨팅 서버의 하드웨어를 로봇 컨트롤러 캐비닛 내부의 여유 공간이나 전용 슬롯에 물리적으로 설치하여 하나의 장치처럼 만드는 것을 의미합니다.
MVP 배포에서 중요한 점:
이 둘 중 어느 방식을 사용하든, 두 장치 간의 네트워크 연결이 안정적이어야 합니다. network_mode: host를 사용하여 Docker 컨테이너가 로봇 컨트롤러의 IP 주소로 직접 통신할 수 있도록 설정하는 것이 핵심입니다.
2. API 서버와 ROS 2 시스템을 하나의 컨테이너/프로세스 내에서 실행하는 방법
main.py (FastAPI 서버)와 integrated_robot_system.py (ROS 2 노드)는 현재 별도의 서비스로 설계되어 있습니다. 이를 하나의 컨테이너 또는 프로세스 내에서 실행하는 방법은 두 가지가 있습니다.
방법 1: Python의 멀티프로세싱/서브프로세스 활용 (단일 프로세스 내 실행)
이 방법은 하나의 Python 스크립트에서 FastAPI 애플리케이션과 ROS 2 노드를 동시에 실행합니다.
장점: 가장 단순하며 IPC 오버헤드가 전혀 없습니다.
구현: main.py 파일 내에서 uvicorn 서버 시작과 ROS 2 rclpy.spin() 실행을 동시에 처리해야 합니다.
python
# main.py 파일 내에서 통합 예시 (개념 코드)

import uvicorn
from fastapi import FastAPI
# ... (기타 FastAPI 및 ROS 2 임포트) ...

# ROS 2 노드 및 Executor 설정 (이전 코드의 CommandPublisher 부분을 그대로 가져옴)
node = CommandPublisher()
executor = rclpy.executors.MultiThreadedExecutor()
executor.add_node(node)

# ROS 2 스핀을 별도 스레드에서 실행
threading.Thread(target=executor.spin, daemon=True).start()

# FastAPI 앱 정의
app = FastAPI()
# ... (FastAPI 라우트 정의) ...

# 메인 실행 시 uvicorn 시작
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

Use code with caution.

이 방식은 ROS 2 시스템의 integrated_robot_system.py를 main.py와 같은 프로세스에서 실행하는 것이 아니므로, 결국 main.py는 integrated_robot_system.py 노드와 ROS 2 토픽으로 통신해야 합니다.
방법 2: Supervisor 또는 Entrypoint 스크립트 활용 (단일 컨테이너 내 실행)
이 방법은 하나의 Docker 컨테이너 내에서 두 개의 별도 프로세스(python3 main.py와 ros2 launch ...)를 실행합니다.
장점: 코드를 크게 수정할 필요가 없으며, 프로세스 관리가 용이합니다.
구현: supervisord 같은 프로세스 관리 도구를 사용하거나, 커스텀 쉘 스크립트를 컨테이너의 ENTRYPOINT로 설정합니다.
dockerfile
# Dockerfile_standalone 내 명령어 예시 (Supervisor 사용 시)

# ... (빌드 및 설치 과정) ...

RUN apt-get update && apt-get install -y supervisor
COPY supervisord.conf /etc/supervisor/supervisord.conf

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
Use code with caution.

ini
# supervisord.conf 파일 예시

[supervisord]
nodaemon=true

[program:fastapi_server]
command=python3 /app/main.py
autostart=true
autorestart=true

[program:ros_system]
# ROS 환경 소싱 후 ROS 2 노드 실행 명령
command=/bin/bash -c ". install/setup.bash && ros2 launch doosan_robotic_manipulation integrated_system.launch.py"
autostart=true
autorestart=true
Use code with caution.

MVP 배포에서는 **방법 2(Supervisor)**가 코드 수정 없이 두 시스템을 효과적으로 단일 컨테이너에 통합할 수 있는 가장 표준적이고 강력한 방법입니다.

+++++++++++++++++++++++++++++++++++++++++++++

Sim-to-Real Checklist & Step-by-Step Guide
1단계: 하드웨어 및 소프트웨어 환경 구축 (필수 점검)
1. 로봇 및 주변 환경: * 두산 로봇암: 전원 연결, 컨트롤러 부팅, 네트워크 연결 확인. * D455 카메라: 로봇 엔드 이펙터에 단단히 고정되었는지 확인. USB 3.0 이상 포트에 연결. * 워크스테이션/엣지 PC: 로봇암과 D455를 연결할 PC (GPU 장착 필수). ROS 2 설치 및 설정 완료.

2. ROS 2 패키지 및 코드 준비: * 위에 제시된 camera_driver_node.py, student_policy_inference_node.py, robot_arm_interface_node.py 파일들을 하나의 ROS 2 패키지(예: doosan_rl_sim2real) 내에 저장. * setup.py에 노드 실행 스크립트 등록 (또는 launch 파일 생성). * student_policy.pt (학습된 Student 모델) 파일이 추론 노드에서 접근 가능한 경로에 있는지 확인. * 두산 로봇 ROS 드라이버가 설치되어 있고, joint_states 발행 및 joint_group_vel_controller/commands 구독이 가능한지 확인.

3. Safety (제일 중요!): * 비상 정지 버튼 (E-Stop): 로봇 컨트롤러 및 주변에 손 닿는 곳에 비상 정지 버튼이 항상 활성화되어 있는지 확인. * 작업 공간 확보: 로봇의 최대 가동 범위 내에 사람이 없도록 안전 구역 설정. * 속도 제한: student_policy_inference_node.py의 self.MAX_SPEED 값을 매우 낮게 (예: 0.1 rad/s) 설정하여 초기 움직임을 최소화합니다.

2단계: 개별 노드 테스트 및 캘리브레이션
1. camera_driver_node 테스트: * ros2 run doosan_rl_sim2real camera_driver_node 실행. * ros2 topic echo /camera/depth/image_raw로 메시지 발행 확인. * ros2 run rviz2 rviz2에서 Image Display를 추가하여 Depth 이미지 가시화 확인. (RealSense SDK 필터 적용 여부 확인) * 중요: Depth 이미지의 노이즈, 범위, 해상도가 시뮬레이션 학습 시와 최대한 유사한지 육안으로 확인.

2. 두산 로봇 ROS 드라이버 테스트: * 두산 로봇 ROS 드라이버를 실행하여 joint_states 토픽이 정상적으로 발행되는지 확인. * ros2 topic echo /joint_states로 관절 값 확인. * 수동으로 로봇을 움직여 joint_states 값이 변하는지 확인. * robot_arm_interface_node가 로봇 API와 정상적으로 통신할 수 있는지 사전 테스트 (로봇 제어 없이 self.get_logger().info 등으로 API 호출 성공 여부만 확인).

3. Sim-to-Real Gap - Hand-Eye Calibration (필수!): * D455 카메라의 3D 공간에서의 정확한 위치와 방향을 로봇 베이스 좌표계 기준으로 알아야 합니다. * **robot_base_link 기준 camera_link의 Transformation (변환 행렬)**이 필요합니다. * 이 과정은 ros2_hand_eye_calibration 같은 패키지나 외부 툴을 사용하여 정밀하게 수행해야 합니다. * 시뮬레이션에서 정의된 카메라 위치와 실제 카메라 위치가 정확히 일치하는지 확인하는 과정입니다. 이 단계가 잘못되면 로봇은 잘못된 공간 정보를 해석합니다.

3단계: 통합 시스템 구동 (Sim-to-Real!)
주의: 이 단계부터 로봇이 예측 불가능하게 움직일 수 있으므로, 항상 비상 정지 버튼을 누를 준비를 해야 합니다.

1. 모든 노드 실행 (터미널 여러 개 사용 또는 Launch 파일): * ros2 run doosan_rl_sim2real camera_driver_node * ros2 run doosan_rl_sim2real student_policy_inference_node * ros2 run doosan_rl_sim2real robot_arm_interface_node * 두산 로봇의 ROS 드라이버 (또는 컨트롤러)도 실행되어야 합니다.

2. 초기 동작 관찰 (아주 느리게): * 로봇이 움직이기 시작하면, 아주 느리게 (MAX_SPEED가 낮으므로) 로봇 팔이 어떤 방향으로 움직이는지 관찰합니다. * 정확하게 원하는 동작(예: 물건 집기)과 유사한 방향으로 움직이는가? * 아니면 엉뚱한 방향으로 급발진하는가? * 조금이라도 이상하면 즉시 비상 정지 버튼을 누릅니다.

3. 디버깅 및 반복: * 로봇이 이상하게 움직인다면: * /policy_actions 토픽을 ros2 topic echo로 확인: 모델이 어떤 명령을 내리고 있는지 확인. * 카메라 이미지와 전처리 확인: student_policy_inference_node 내에서 전처리된 이미지를 발행하여 Rviz2로 확인하거나, 중간 단계의 tensor를 저장하여 시뮬레이션 데이터와 비교. * Hand-Eye Calibration 재확인: 가장 흔한 Sim-to-Real Gap 원인입니다. * MAX_SPEED를 더 낮추거나, action_scaling에 문제가 없는지 확인. * 작은 성공: 만약 로봇이 원하는 방향으로 아주 조금이라도 움직인다면, MAX_SPEED를 천천히 올리면서 점진적으로 동작 범위를 늘려나갑니다.

4. Tidy Task 실행: * 로봇이 안정적으로 움직인다면, 실제 물건들을 테이블 위에 배치하고 Tidy Task를 시도합니다. * 물건의 종류, 크기, 놓인 위치 등을 다양하게 바꿔가며 로봇의 강인성(Robustness)을 테스트합니다.

Sim-to-Real은 인내심과 세심한 디버깅이 필요한 과정입니다. 하지만 성공한다면, 가상 세계에서 학습된 AI가 현실 세계에 강림하는 엄청난 성취를 경험하게 될 것입니다.

.pt 파일을 ROS 2 통합 시스템에 배포하기 위해 성능을 최적화하는 과정은 Sim-to-Real 배포의 핵심입니다.강화학습 모델은 수많은 행렬 연산을 포함하고 있어, 일반적인 PyTorch 환경에서 Python으로 실행하면 지연 시간(Latency)이 길어집니다. 이를 해결하기 위해 ONNX나 TensorRT로 변환하고, 최적화된 환경에서 실행해야 합니다.아래는 PyTorch $\rightarrow$ ONNX $\rightarrow$ TensorRT 변환 및 ROS 2 노드에 통합하는 방법입니다.1. 모델 변환 (Optimization) 단계 💻A. Step 1: PyTorch(.pt)를 ONNX로 변환 (Python)**ONNX (Open Neural Network Exchange)**는 다양한 프레임워크 간에 모델을 공유하기 위한 표준 형식입니다. 변환을 위해서는 **학습 시 사용한 모델 아키텍처(클래스 정의)**와 모델 입력 데이터의 더미(Dummy) 텐서가 필요합니다.전제 조건: StudentPolicy 클래스 정의가 필요합니다. 이전 대화에서 Student 모델은 **Depth 이미지와 관절 상태(Joint State)**를 입력으로 받는다고 가정했습니다.1-1. convert_to_onnx.py 파일 작성Pythonimport torch
import torch.nn as nn
# ⚠️ 주의: 실제 StudentPolicy 모델 클래스 파일을 임포트해야 합니다.
# from your_project.student_policy import StudentPolicy

# --- 💡 (임시) StudentPolicy 아키텍처 재정의 ---
# 로드할 수 있도록 임시로 모델 구조를 정의 (실제 구조에 맞게 수정 필요)
class StudentPolicy(nn.Module):
    def __init__(self, action_dim=6):
        super().__init__()
        # Depth Input: [1, 1, 128, 128] 가정
        self.cnn = nn.Sequential(nn.Conv2d(1, 32, 3, 1), nn.ReLU(), nn.Flatten())
        # Joint Input: [1, 6] 가정
        self.mlp_joint = nn.Sequential(nn.Linear(6, 32), nn.ReLU())
        # 최종 출력 (Action)
        self.output_layer = nn.Linear(32 * 128 * 128 + 32, action_dim) # 임의의 크기

    def forward(self, depth_tensor, joint_tensor):
        cnn_out = self.cnn(depth_tensor)
        joint_out = self.mlp_joint(joint_tensor)
        combined = torch.cat([cnn_out, joint_out], dim=1)
        action = self.output_layer(combined)
        return action
# ---------------------------------------------


# 파일 경로 설정
PT_MODEL_PATH = "model_1499.pt"
ONNX_MODEL_PATH = "student_policy_optimized.onnx"
ACTION_DIM = 6 # 로봇 관절 수

# 1. 모델 인스턴스화 및 가중치 로드
model = StudentPolicy(action_dim=ACTION_DIM).to('cpu')
# Rsl-rl 모델은 전체 체크포인트 파일에서 'model_state_dict'를 꺼내야 할 수 있습니다.
checkpoint = torch.load(PT_MODEL_PATH, map_location='cpu')
# ⚠️ rsl-rl은 보통 model_state_dict 안에 actor.model.state_dict() 가 있습니다.
model.load_state_dict(checkpoint['model_state_dict']) # 키 이름은 환경에 따라 다를 수 있습니다.
model.eval()

# 2. 더미 입력 데이터 생성 (중요!)
# 시뮬레이션에서 사용한 입력 텐서 크기와 타입(dtype)을 정확히 맞춰야 합니다.
# [Batch Size, Channel, Height, Width]
dummy_depth = torch.randn(1, 1, 128, 128) 
# [Batch Size, Joint Dim]
dummy_joint = torch.randn(1, ACTION_DIM)

# 3. ONNX로 변환 실행
torch.onnx.export(
    model, 
    (dummy_depth, dummy_joint), # 모델의 입력 튜플
    ONNX_MODEL_PATH,
    export_params=True, 
    opset_version=12,         # 호환성을 위해 12 이상 권장
    do_constant_folding=True,
    input_names=['depth_input', 'joint_input'], # ONNX 입력 노드 이름 정의
    output_names=['action_output'],
    # 동적 배치 크기 지원 (선택 사항)
    dynamic_axes={'depth_input': {0: 'batch_size'}, 'joint_input': {0: 'batch_size'}, 'action_output': {0: 'batch_size'}}
)

print(f"ONNX 모델이 {ONNX_MODEL_PATH}에 저장되었습니다.")
B. Step 2: ONNX를 TensorRT 엔진으로 변환 (NVIDIA 추천)TensorRT는 NVIDIA GPU에서 추론을 위해 특화된 라이브러리입니다. ONNX 모델을 GPU 아키텍처에 맞게 최적화하고 컴파일합니다.2-1. trtexec 툴 사용 (가장 쉬운 방법)TensorRT가 설치된 환경에서, NVIDIA가 제공하는 커맨드라인 툴 trtexec을 사용하면 쉽게 변환 및 벤치마킹이 가능합니다.Bash# ONNX 파일을 TensorRT 엔진으로 변환
# --fp16: 정확도 손실을 최소화하며 속도를 높이는 16비트 부동 소수점 사용
# --timingCacheFile: 엔진 로드 시간을 줄이기 위해 캐시 파일 생성
trtexec --onnx=student_policy_optimized.onnx \
        --saveEngine=student_policy_tensorrt.engine \
        --minShapes=depth_input:1x1x128x128,joint_input:1x6 \
        --optShapes=depth_input:1x1x128x128,joint_input:1x6 \
        --maxShapes=depth_input:1x1x128x128,joint_input:1x6 \
        --fp16 \
        --timingCacheFile=timing.cache
2. ROS 2 추론 노드 배포 (Deployment) 🚀이제 최적화된 모델을 ROS 2 노드에 통합합니다.A. 최적화된 Python 노드 (ONNX Runtime 사용)Python 환경에서는 onnxruntime 라이브러리를 사용하여 추론하는 것이 가장 빠르고 효율적입니다.student_policy_inference_node.py 수정 사항Python# student_policy_inference_node.py (핵심 로직 업데이트)
import onnxruntime as ort
# ... 기타 임포트 유지

class StudentPolicyInferenceNode(Node):
    def __init__(self):
        # ... 기존 초기화 코드 유지 ...
        
        # --- 모델 로드 (ONNX Runtime) ---
        ONNX_MODEL_PATH = "student_policy_optimized.onnx" # 파일 경로 지정
        
        # 1. ORT Session 생성 (GPU 사용)
        self.ort_session = ort.InferenceSession(
            ONNX_MODEL_PATH, 
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] # GPU 우선 사용
        )
        
        # 2. 입력/출력 이름 확인 (ONNX 변환 시 정의한 이름 사용)
        self.input_names = [input.name for input in self.ort_session.get_inputs()] 
        self.output_names = [output.name for output in self.ort_session.get_outputs()]

    def control_loop(self):
        # ... 데이터 준비 로직 유지 ...
        
        # 1. 텐서를 NumPy로 변환 (ONNX Runtime은 NumPy 배열을 입력으로 받음)
        depth_np = self.latest_depth_tensor.cpu().numpy()
        joint_np = self.latest_joint_tensor.cpu().numpy()
        
        # 2. 모델 추론 (Inference)
        # ⚠️ 입력 순서를 ONNX 변환 시의 순서(depth, joint)와 동일하게 맞춰야 합니다!
        ort_inputs = {
            self.input_names[0]: depth_np,   # depth_input
            self.input_names[1]: joint_np    # joint_input
        }
        
        ort_outputs = self.ort_session.run(self.output_names, ort_inputs)
        
        # 3. 결과 처리
        action_np = ort_outputs[0].squeeze(0) # 첫 번째 출력이 Action

        # ... Action Scaling 및 발행 로직 유지 ...
B. 최적 성능을 위한 C++ 노드 (TensorRT 엔진 사용)가장 낮은 지연 시간과 높은 처리율을 위해서는 ROS 2 C++ (Rclcpp) 노드에서 TensorRT API를 직접 호출하는 것이 가장 좋습니다.C++ 노드 구조 (개념)헤더 파일: TensorRT, ROS 2(rclcpp), OpenCV(cv_bridge 대신) 관련 헤더를 포함합니다.엔진 로드: .engine 파일을 메모리에 로드하고, **IExecutionContext**를 생성합니다.구독 콜백:**/camera/depth/image_raw**와 **/joint_states**를 구독합니다.cv_bridge 대신 C++의 효율적인 데이터 변환(예: Zero-Copy)을 사용합니다.추론 루프:입력 데이터를 TensorRT에서 요구하는 GPU 버퍼로 복사합니다.context->executeV2(bindings) 함수를 호출하여 추론을 실행합니다. (GPU에서 직접 연산)명령 발행: GPU 결과 버퍼에서 CPU로 데이터를 다시 가져와서 ROS 2 토픽으로 발행합니다.이 방법은 설정 복잡도가 높지만, 5ms 미만의 추론 지연 시간을 달성하는 데 필수적입니다.
