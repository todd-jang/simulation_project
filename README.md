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
