/*
 * =====================================================================================
 *
 * 파일 이름:  main_control_node.cpp
 *
 * 설명: Franka Panda 로봇을 위한 메인 제어 노드 (C++)
 * - AI (DQN) 노드로부터 관절 명령을 구독 (Subscription)
 * - Franka의 ros2_control 컨트롤러로 실제 궤적 명령을 발행 (Publication)
 *
 * =====================================================================================
 */

#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"

// Franka Panda의 7개 관절 이름을 정의합니다.
const std::vector<std::string> PANDA_JOINTS = {
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7"
};

class MainControlNode : public rclcpp::Node
{
public:
  MainControlNode() : Node("main_control_node")
  {
    RCLCPP_INFO(this->get_logger(), "메인 제어 노드(C++) 초기화 중...");

    // AI (DQN) 노드로부터 고수준 관절 명령을 구독
    ai_command_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
        "/panda_ai_commands", // DQN 노드가 발행하는 토픽
        10,
        std::bind(&MainControlNode::aiCommandCallback, this, std::placeholders::_1));

    // Franka의 ros2_control 컨트롤러로 궤적 메시지를 발행
    trajectory_pub_ = this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
        // "/joint_trajectory_controller/joint_trajectory" (실제 환경에 맞게 수정 필요)
        "/panda_arm_controller/joint_trajectory",
        10);
    
    RCLCPP_INFO(this->get_logger(), "메인 제어 노드 준비 완료. AI 명령 대기 중...");
  }

private:
  void aiCommandCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
  {
    if (msg->data.size() != PANDA_JOINTS.size())
    {
      RCLCPP_ERROR(this->get_logger(), "DQN 명령의 관절 수가 7개가 아닙니다! (수신: %zu)", msg->data.size());
      return;
    }

    RCLCPP_DEBUG(this->get_logger(), "AI로부터 새 관절 명령 수신");

    // trajectory_msgs::msg::JointTrajectory 메시지 생성
    auto trajectory_msg = std::make_unique<trajectory_msgs::msg::JointTrajectory>();
    trajectory_msg->header.stamp = this->get_clock()->now();
    trajectory_msg->joint_names = PANDA_JOINTS;

    // 궤적 포인트 생성 (AI 명령을 0.5초 안에 도달하도록 설정)
    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.positions = msg->data;
    point.time_from_start = rclcpp::Duration(0, 500000000); // 0.5초

    trajectory_msg->points.push_back(point);

    // 실제 로봇 컨트롤러로 궤적 발행
    trajectory_pub_->publish(std::move(trajectory_msg));
  }

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr ai_command_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_pub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  // MainControlNode를 생성하고 실행합니다.
  rclcpp::spin(std::make_shared<MainControlNode>());
  rclcpp::shutdown();
  return 0;
}
