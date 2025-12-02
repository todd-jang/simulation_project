cmake_minimum_required(VERSION 3.8)
project(doosan_rl_sim2real) # 패키지 이름

# ... 나머지 설정 유지 ...

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

# --- TensorRT/CUDA 설정 (핵심) ---
# 일반적으로 TensorRT는 CUDA 설치 경로 내에 포함되어 있습니다.
# 사용자 환경에 따라 CUDA_TOOLKIT_ROOT_DIR 경로를 설정해야 할 수 있습니다.

find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED) # TensorRT는 CMake 파일 제공이 없을 수 있습니다. 

# ⚠️ TensorRT가 find_package로 찾아지지 않는 경우, 직접 경로를 지정해야 합니다.
# 예시: set(TensorRT_ROOT "/usr/src/tensorrt")
#       include_directories(${TensorRT_ROOT}/include)

include_directories(
  include
  ${TensorRT_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(tensorrt_inference_node src/tensorrt_inference_node.cpp)

ament_target_dependencies(tensorrt_inference_node 
  rclcpp 
  sensor_msgs 
  std_msgs 
  cv_bridge
  image_transport
)

target_link_libraries(tensorrt_inference_node
  ${OpenCV_LIBS}
  ${TensorRT_LIBS} # libnvinfer.so 등
  ${CUDA_LIBRARIES}
)

install(TARGETS
  tensorrt_inference_node
  DESTINATION lib/${PROJECT_NAME}
)


cmake_minimum_required(VERSION 3.8)
project(panda_ai_control)

# ... (기존 find_package 동일) ...
find_package(ament_cmake REQUIRED)
# ...

# ... (기존 C++ 실행 파일 동일) ...
add_executable(
  main_control_node
  src/main_control_node.cpp
)
# ...

# ... (기존 C++ 설치 동일) ...
install(TARGETS
  main_control_node
  DESTINATION lib/${PROJECT_NAME}
)

# Python 스크립트 설치
install(
  PROGRAMS
  "scripts/bev_vision_node.py"
  "scripts/pose_estimation_node.py"
  "scripts/dqn_control_node.py"
  "scripts/task_orchestrator_node.py" # <-- [!!!] 이 줄 추가
  DESTINATION lib/${PROJECT_NAME}
)

# ... (기존 launch 및 config 설치 동일) ...
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# ... (기존 테스트 동일) ...

ament_package(
