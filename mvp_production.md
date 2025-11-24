#devops

이번 MVP(Minimum Viable Product)를 독립형(standalone) 로봇 시스템에 배포한다면, **"기능 검증"과 "최소한의 안전성 확보"**에 초점을 맞춰야 합니다. 목표는 프론트엔드와 클라우드 의존성을 제거하고, 핵심 제어 로직을 로봇 시스템 자체에서 실행하는 것입니다.
다음은 독립형 배포를 위한 핵심 단계와 변경사항입니다.
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

 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 시스템 장점 

최적화된 ROS 2 패키지 구조 (doosan_hri_project)
워크스페이스 이름은 doosan_hri_project로 명명하며, src 폴더 내의 각 패키지와 핵심 노드는 다음과 같이 구성됩니다.
패키지 이름	유형	주요 역할 및 포함 노드
dsr_bringup	ROS pkg (Python/XML)	시스템 실행 및 관리:
- 실제 로봇 및 Isaac Sim 시뮬레이션 환경 실행 Launch 파일 포함.
- 전체 시스템 (vision_pipeline, manipulation_planning)을 시작하는 마스터 런치 파일 (system_bringup.launch.py).
dsr_description	ROS pkg (XML/URDF)	로봇 모델 정의:
- 두산 E0509 및 로보티즈 2지 그리퍼의 URDF/XACRO 모델.
- MoveIt! 설정 패키지 (dsr_moveit_config)를 위한 설정 파일.
- Isaac Lab에서 사용할 USD 모델 파일 관리.
vision_pipeline	ROS pkg (Python)	지능형 비전 처리:
- object_detection_node: 뎁스 카메라 데이터를 기반으로 고정된 물체(상자 등)의 6D Pose를 추정하고 발행 (/object_detection/pose). YOLO 사용.
- dynamic_tracking_node: OpenCV/YOLO를 사용하여 동적 객체(사람 손) 추적 및 위치 발행 (/dynamic_tracking/pose).
manipulation_planning	ROS pkg (Python)	동작 제어 및 RL 통합 (핵심):
- integrated_robot_system.py (메인 노드): 시스템의 중추.
- MoveIt!을 초기 설정 및 폴백(fallback)으로 사용.
- Isaac Lab에서 학습된 RL 정책 모델(.pth) 로드 및 추론 실행.
- 비전 및 HRI 데이터를 바탕으로 로봇 제어 명령 생성 및 실행.
- 태스크 에러 시 재시도 및 홈 대기 로직 포함.
hri_interface	ROS pkg (Python)	인간-로봇 상호작용 인터페이스:
- language_interface_node: FastAPI 백엔드로부터 JSON 명령 (/hri/command_json)을 구독하고, 이를 로봇 컨트롤러(integrated_robot_system)가 이해하는 내부 목표 상태로 변환.
시스템 간 데이터 흐름 요약
HRI (프론트엔드/FastAPI) -> ROS:
FastAPI 서버가 hri_interface의 /hri/command_json 토픽으로 JSON 명령을 발행합니다.
비전 -> ROS:
vision_pipeline은 /camera/... 토픽을 구독하여 /object_detection/pose를 발행합니다.
제어 (메인 로직):
manipulation_planning의 integrated_robot_system 노드는 HRI 명령과 비전 Pose를 모두 구독합니다.
수신된 명령에 따라 RL 모델 추론을 실행하거나, MoveIt! 폴백을 사용하여 로봇 조인트 명령 토픽(joint_trajectory_controller/...)으로 발행합니다.
이 구조는 독립 실행형(standalone) 배포와 클라우드 연동 배포 모두에 유연하게 대응하며, 사용자의 MVP 목표에 완벽하게 부합합니다
