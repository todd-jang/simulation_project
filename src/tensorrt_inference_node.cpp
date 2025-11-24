#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

// TensorRT 헤더
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <vector>

// 1. TensorRT 로거 구현 (필수)
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // ROS 2 로거와 연결
        if (severity <= Severity::kERROR) {
            RCLCPP_ERROR(rclcpp::get_logger("TensorRT"), "TRT Error: %s", msg);
        } else if (severity == Severity::kWARNING) {
            RCLCPP_WARN(rclcpp::get_logger("TensorRT"), "TRT Warning: %s", msg);
        } else {
            RCLCPP_INFO(rclcpp::get_logger("TensorRT"), "TRT Info: %s", msg);
        }
    }
};

// ... (TensorRT 유틸리티 함수: CudaMallocWrapper, CudaFreeWrapper 등) ...


class TensorRTInferenceNode : public rclcpp::Node {
public:
    TensorRTInferenceNode() : Node("tensorrt_inference_node") {
        // 파라미터 선언 및 로드
        this->declare_parameter<std::string>("model_engine_path", "");
        std::string engine_path = this->get_parameter("model_engine_path").as_string();

        // 1. TensorRT 엔진 로드
        if (!loadEngine(engine_path)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load TensorRT engine!");
            rclcpp::shutdown();
            return;
        }

        // 2. ROS 2 구독/발행 설정
        depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/depth/image_raw", 10, std::bind(&TensorRTInferenceNode::depthCallback, this, std::placeholders::_1));
        
        joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10, std::bind(&TensorRTInferenceNode::jointCallback, this, std::placeholders::_1));
        
        action_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/policy_actions", 10);
        
        // 3. 제어 타이머 설정 (High Frequency)
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20), // 50 Hz
            std::bind(&TensorRTInferenceNode::inferenceLoop, this));
        
        RCLCPP_INFO(this->get_logger(), "TensorRT Inference Node 초기화 완료.");
    }

    ~TensorRTInferenceNode() {
        // CUDA 메모리 해제
        if (context_) context_->destroy();
        if (engine_) engine_->destroy();
        for (void* buffer : bindings_) cudaFree(buffer);
        cudaStreamDestroy(stream_);
    }

private:
    // TensorRT 객체
    Logger logger_;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    std::vector<void*> bindings_; // GPU 메모리 포인터
    
    // ROS 2 객체
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr action_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // 데이터 저장
    std::vector<float> latest_joint_data_;
    std::vector<float> latest_depth_data_;
    std::mutex data_mutex_;

    // [이미지 로드 및 전처리 로직]
    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        // C++ OpenCV(cv::Mat)를 사용하여 이미지 전처리
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
            cv::Mat depth_img = cv_ptr->image;

            // 1. 리사이징 (예: 128x128)
            cv::resize(depth_img, depth_img, cv::Size(128, 128));

            // 2. 정규화 및 float 변환 (학습 시와 동일하게!)
            depth_img.convertTo(depth_img, CV_32FC1, 1.0/5000.0); // 5000.0은 최대 depth 값

            // 3. 벡터로 복사 (GPU 전송 준비)
            std::lock_guard<std::mutex> lock(data_mutex_);
            latest_depth_data_.assign((float*)depth_img.data, (float*)depth_img.data + depth_img.total());

        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }
    
    // [관절 데이터 저장 로직]
    void jointCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        // 관절 위치/속도 등 학습에 사용된 데이터를 추출
        latest_joint_data_ = msg->position; 
    }

    // [TensorRT 엔진 로드 로직]
    bool loadEngine(const std::string& path) {
        // ... 파일에서 엔진을 읽고, Deserialize하고, Context를 생성하는 표준 TensorRT C++ 로직 구현 ...
        // 이 과정에서 input/output binding 정보를 바탕으로 GPU 메모리를 할당합니다 (cudaMalloc).
        // input: depth_input (1x1x128x128), joint_input (1x6)
        // output: action_output (1x6)
        return true; 
    }

    // [추론 및 명령 발행 루프]
    void inferenceLoop() {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (latest_depth_data_.empty() || latest_joint_data_.size() != 6) {
            return;
        }

        // 1. CPU 데이터 -> GPU 메모리로 비동기 복사 (CUDA Stream 활용)
        // cudaMemcpyAsync(bindings_[input_idx_depth], latest_depth_data_.data(), size_depth, cudaMemcpyHostToDevice, stream_);
        // cudaMemcpyAsync(bindings_[input_idx_joint], latest_joint_data_.data(), size_joint, cudaMemcpyHostToDevice, stream_);

        // 2. 추론 실행
        // context_->enqueueV2(bindings_.data(), stream_, nullptr); // 비동기 실행

        // 3. GPU 결과 -> CPU 메모리로 비동기 복사 (Action)
        std::vector<float> action_output(6);
        // cudaMemcpyAsync(action_output.data(), bindings_[output_idx_action], size_action, cudaMemcpyDeviceToHost, stream_);
        
        // 4. 스트림 동기화 (결과를 사용하기 전에 반드시 완료되어야 함)
        // cudaStreamSynchronize(stream_); 

        // 5. ROS 2 명령 발행
        auto msg = std::make_unique<std_msgs::msg::Float64MultiArray>();
        msg->data = action_output;
        action_pub_->publish(std::move(msg));
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TensorRTInferenceNode>());
    rclcpp::shutdown();
    return 0;
}

