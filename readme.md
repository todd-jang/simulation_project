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



++++++++++++++++++++++++++++++++


그림처럼 rosbridge_server는 웹(Frontend)과 ROS 2(Backend) 사이의 다리 역할을 합니다.



다음 단계를 따라 설치하고 실행할 수 있습니다.

1. 설치 (Installation)

rosbridge_server는 ROS 2의 공식 패키지이므로 apt를 통해 간단히 설치할 수 있습니다.

새 터미널을 여시고 다음 명령어를 입력하세요.

Bash
# 1. 패키지 리스트 업데이트
sudo apt update

# 2. rosbridge_server 설치
# [중요!] 본인의 ROS 2 버전에 맞게 'humble' 부분을 수정하세요.
# (예: foxy, galactic, humble, iron 등)
sudo apt install ros-humble-rosbridge-server
2. 실행 (Execution)

rosbridge_server는 panda_ai_control 런치 파일과는 별개의 터미널에서 실행되어야 합니다.

새 터미널을 엽니다.

ROS 2 환경을 소싱(source)합니다.

Bash
# [중요!] 본인의 ROS 2 버전에 맞게 'humble'을 수정하세요.
source /opt/ros/humble/setup.bash

# (만약 'panda_ai_control' 워크스페이스도 소싱했다면 더 좋습니다)
# source ~/ros2_ws/install/setup.bash 
ros2 launch 명령어로 rosbridge_server의 기본 웹소켓 런치 파일을 실행합니다.

Bash
ros2 launch rosbridge_server rosbridge_websocket_launch.py
이 명령어를 실행하면, 터미널에 Rosbridge WebSocket server started on port 9090와 같은 메시지가 나타나며 서버가 시작됩니다.


=======================================================

이제 시스템을 실행하려면 총 2개의 터미널이 필요합니다.

터미널 1 (AI 두뇌):

Bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch panda_ai_control start_robot_llm.launch.py


터미널 2 (웹 관문):

Bash
source /opt/ros/humble/setup.bash
ros2 launch rosbridge_server rosbridge_websocket_launch.py
이제 프론트엔드(웹 앱)는 ws://<로봇의_IP_주소>:9090 주소로 WebSocket 연결을 시도할 수 있습니다. 연결이 성공하면, 웹 앱은 /user_command 토픽으로 ASR 텍스트를 발행(publish)할 수 있게 됩니다.
