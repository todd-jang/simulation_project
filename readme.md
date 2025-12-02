TensorRT C++ API를 사용하는 ROS 2 C++ 노드는 **최고의 추론 속도와 가장 낮은 지연 시간(Low Latency)**을 달성하기 위한 표준적인 방법입니다. C++을 사용하면 Python의 오버헤드를 완전히 제거하고, GPU 메모리(CUDA)를 직접 관리하여 데이터를 복사 없이 빠르게 처리할 수 있습니다.



빌드 및 실행 방법

파일 저장: 위의 8개 파일을 ~/ros2_ws/src/ 디렉터리 아래에 panda_ai_control 패키지 디렉터리를 만들어 정확한 경로에 저장합니다. (예: ~/ros2_ws/src/panda_ai_control/package.xml)

패키지 빌드: ROS 2 작업 공간 루트(예: ~/ros2_ws)에서 빌드합니다.

Bash
cd ~/ros2_ws
colcon build --packages-select panda_ai_control

환경 설정:

Bash
source ~/ros2_ws/install/setup.bash

시스템 실행: 아래의 런치 명령어를 실행하면 4개의 노드(C++ 1개, Python 3개)가 모두 실행되며 AI 기반 로봇 제어 시스템이 가동됩니다.

Bash
ros2 launch panda_ai_control start_robot.launch.py

이 패키지는 실제 AI 모델(DQN, 자세 추정)과 BEV 알고리즘 로직이 **Placeholder(자리표시자)**로 구현되어 있지만, 사용자가 요청한 모든 구성 요소를 통합하는 완전하고 빌드 가능한 ROS 2 패키지 구조를 제공합니다.
