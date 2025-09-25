
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <algorithm>
#include <numeric>
#include <optional>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"

// PCL specific includes
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>

// Open3D for DBSCAN
#include <open3d/Open3D.h>

// --- NEW: ONNX Runtime for ML model inference ---
#include <onnxruntime_cxx_api.h>

using PointT = pcl::PointXYZI;

class DataExtractorWithMarkers : public rclcpp::Node
{
public:
    DataExtractorWithMarkers() 
    : Node("data_extractor_with_markers"), env_(ORT_LOGGING_LEVEL_WARNING, "cone-detector")
    {
        // General parameters
        this->declare_parameter<std::string>("ground_removal_method", "ransac");
        this->declare_parameter<double>("max_queue_size", 1.0);
        this->declare_parameter<std::string>("onnx_model_path", "cone_model.onnx");

        // Filter parameters
        this->declare_parameter<double>("y_filter_min", -4.0);
        this->declare_parameter<double>("y_filter_max", 4.0);
        this->declare_parameter<double>("x_filter_min", 0.0);
        this->declare_parameter<double>("x_filter_max", 10.0);

        // Clustering parameters
        this->declare_parameter<double>("dbscan_eps", 0.3);
        this->declare_parameter<int>("dbscan_min_points", 5);

        // Ground removal parameters
        this->declare_parameter<double>("ground_height_threshold", -0.1627);
        this->declare_parameter<double>("ransac_distance_threshold", 0.05);
        this->declare_parameter<int>("ransac_max_iterations", 5);
        this->declare_parameter<double>("ransac_min_z_normal", 0.90);
        this->declare_parameter<double>("ransac_max_slope_deviation", 10.0);

        // Get all parameters
        this->get_parameter("ground_removal_method", ground_removal_method_);
        this->get_parameter("y_filter_min", y_filter_min_);
        this->get_parameter("y_filter_max", y_filter_max_);
        this->get_parameter("x_filter_min", x_filter_min_);
        this->get_parameter("x_filter_max", x_filter_max_);
        this->get_parameter("dbscan_eps", dbscan_eps_);
        this->get_parameter("dbscan_min_points", dbscan_min_points_);
        this->get_parameter("ground_height_threshold", ground_height_threshold_);
        this->get_parameter("ransac_distance_threshold", ransac_distance_threshold_);
        this->get_parameter("ransac_max_iterations", ransac_max_iterations_);
        this->get_parameter("ransac_min_z_normal", ransac_min_z_normal_);
        this->get_parameter("ransac_max_slope_deviation", ransac_max_slope_deviation_);
        this->get_parameter("onnx_model_path", onnx_model_path_);
        
        double max_queue_size;
        this->get_parameter("max_queue_size", max_queue_size);
        max_queue_size_ = static_cast<size_t>(max_queue_size);

        // --- NEW: Initialize the ONNX Runtime session ---
        initialize_onnx();

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud>(
            "/carmaker/pointcloud",
            rclcpp::SensorDataQoS(),
            std::bind(&DataExtractorWithMarkers::lidar_callback, this, std::placeholders::_1));

        marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/cluster_markers", 10);
        
        RCLCPP_INFO(this->get_logger(), "Data extractor node with ML inference started.");
        
        shutdown_flag_.store(false);
        worker_thread_ = std::thread(&DataExtractorWithMarkers::processing_worker, this);
    }

    ~DataExtractorWithMarkers()
    {
        RCLCPP_INFO(this->get_logger(), "Initiating shutdown...");
        shutdown_flag_.store(true);
        condition_.notify_all();
        
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        RCLCPP_INFO(this->get_logger(), "Shutdown complete.");
    }

private:
    // --- NEW: Function to load the ONNX model ---
    void initialize_onnx() {
        try {
            RCLCPP_INFO(this->get_logger(), "Loading ONNX model from: %s", onnx_model_path_.c_str());
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            session_ = std::make_unique<Ort::Session>(env_, onnx_model_path_.c_str(), session_options);
            
            // Get input and output node names and store them safely in std::string
            Ort::AllocatedStringPtr in_name_ptr = session_->GetInputNameAllocated(0, allocator_);
            Ort::AllocatedStringPtr out_name_ptr = session_->GetOutputNameAllocated(0, allocator_);
            input_name_str_ = std::string(in_name_ptr.get());
            output_name_str_ = std::string(out_name_ptr.get());

            input_node_names_ = { input_name_str_.c_str() };
            output_node_names_ = { output_name_str_.c_str() };

            RCLCPP_INFO(this->get_logger(), "Successfully loaded ONNX model.");
            RCLCPP_INFO(this->get_logger(), "Input node name: %s", input_name_str_.c_str());
            RCLCPP_INFO(this->get_logger(), "Output node name: %s", output_name_str_.c_str());

        } catch (const Ort::Exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Failed to initialize ONNX session: %s", e.what());
            rclcpp::shutdown();
        }
    }
    
    // --- NEW: Function to run inference on a single cluster ---
    int predict_cone_type(const std::vector<float>& features) {
        if (features.size() != NUM_BINS) {
            RCLCPP_ERROR(this->get_logger(), "Feature vector size is %zu, but model expects %d.", features.size(), NUM_BINS);
            return 0; // Default to yellow on error
        }

        // Define the shape of the input tensor: 1 sample, NUM_BINS features
        std::vector<int64_t> input_shape = {1, NUM_BINS};

        // Create a MemoryInfo object for CPU (wraps existing buffer -> zero-copy)
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Create the input tensor by wrapping the existing features buffer.
        // ORT expects a non-const T*, so we const_cast (features is owned by caller).
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(features.data()),
            static_cast<size_t>(features.size()),
            input_shape.data(),
            input_shape.size()
        );

        try {
            // Build an array of input pointers (Run expects pointer to Ort::Value*)
            const Ort::Value* input_ptrs[] = { &input_tensor };

            // Run inference
            std::vector<Ort::Value> input_tensors;
            input_tensors.emplace_back(std::move(input_tensor));

            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_node_names_.data(),
                input_tensors.data(),     // ✅ pointer to Ort::Value
                input_tensors.size(),
                output_node_names_.data(),
                output_node_names_.size()
            );

            int64_t* prediction = output_tensors[0].GetTensorMutableData<int64_t>();
            return static_cast<int>(prediction[0]);

        } catch (const Ort::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "ONNX inference failed: %s", e.what());
            return 0; // Default to yellow on error
        }
    }
    
    void lidar_callback(const sensor_msgs::msg::PointCloud::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (cloud_queue_.size() >= max_queue_size_) {
            cloud_queue_.pop();
        }
        cloud_queue_.push(msg);
        condition_.notify_one();
    }

    void processing_worker()
    {
        while (!shutdown_flag_.load()) {
            sensor_msgs::msg::PointCloud::SharedPtr msg;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                condition_.wait(lock, [this] { return !cloud_queue_.empty() || shutdown_flag_.load(); });
                if (shutdown_flag_.load()) break;
                msg = cloud_queue_.front();
                cloud_queue_.pop();
            }
            if (msg) {
                try {
                    process_cloud(msg);
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "Exception in process_cloud: %s", e.what());
                }
            }
        }
    }

    std::vector<pcl::PointIndices> open3d_dbscan_cluster(const typename pcl::PointCloud<PointT>::Ptr& cloud, double eps, int min_pts)
    {
        if (cloud->points.empty()) return {};
        auto o3d_pcd = std::make_shared<open3d::geometry::PointCloud>();
        o3d_pcd->points_.reserve(cloud->points.size());
        for (const auto& point : cloud->points) {
            o3d_pcd->points_.emplace_back(point.x, point.y, point.z);
        }
        std::vector<int> labels = o3d_pcd->ClusterDBSCAN(eps, min_pts, false);
        if (labels.empty()) return {};
        int max_label = *std::max_element(labels.begin(), labels.end());
        std::vector<pcl::PointIndices> cluster_indices;
        if (max_label >= 0) {
            cluster_indices.resize(max_label + 1);
            for (size_t i = 0; i < labels.size(); ++i) {
                if (labels[i] != -1) {
                    cluster_indices[labels[i]].indices.push_back(i);
                }
            }
        }
        return cluster_indices;
    }

    pcl::PointCloud<PointT>::Ptr remove_ground_by_ransac(const pcl::PointCloud<PointT>::Ptr& cloud)
    {
        pcl::SACSegmentation<PointT> seg;
        pcl::ExtractIndices<PointT> extract;
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(ransac_distance_threshold_);

        auto remaining_cloud = pcl::make_shared<pcl::PointCloud<PointT>>(*cloud);
        std::optional<Eigen::Vector3f> reference_normal;
        int iterations = 0;
        const size_t min_points_for_plane = 100;

        while (remaining_cloud->points.size() > min_points_for_plane && iterations < ransac_max_iterations_)
        {
            seg.setInputCloud(remaining_cloud);
            seg.segment(*inliers, *coefficients);
            if (inliers->indices.empty()) break;

            Eigen::Vector3f current_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
            if (current_normal.z() < 0) current_normal = -current_normal;
            if (current_normal.z() < ransac_min_z_normal_) break;

            if (!reference_normal.has_value()) {
                reference_normal = current_normal;
            } else {
                double angle_rad = std::acos(std::clamp(static_cast<float>(current_normal.dot(reference_normal.value())), -1.0f, 1.0f));
                if (angle_rad * (180.0 / M_PI) > ransac_max_slope_deviation_) break;
            }

            extract.setInputCloud(remaining_cloud);
            extract.setIndices(inliers);
            extract.setNegative(true);
            auto next_remaining_cloud = pcl::make_shared<pcl::PointCloud<PointT>>();
            extract.filter(*next_remaining_cloud);
            remaining_cloud = next_remaining_cloud;
            iterations++;
        }
        return remaining_cloud;
    }

    void process_cloud(const sensor_msgs::msg::PointCloud::SharedPtr msg)
    {
        frame_id_++;
        if (msg->points.empty()) return;

        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
        cloud->points.reserve(msg->points.size());
        for (size_t i = 0; i < msg->points.size(); ++i) {
            PointT point;
            point.x = msg->points[i].x; point.y = msg->points[i].y; point.z = msg->points[i].z;
            point.intensity = (!msg->channels.empty() && i < msg->channels[0].values.size()) ? msg->channels[0].values[i] : 0.0f;
            cloud->points.push_back(point);
        }

        pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
        pcl::PassThrough<PointT> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("y"); pass.setFilterLimits(y_filter_min_, y_filter_max_); pass.filter(*cloud_filtered);
        pass.setInputCloud(cloud_filtered);
        pass.setFilterFieldName("x"); pass.setFilterLimits(x_filter_min_, x_filter_max_); pass.filter(*cloud_filtered);

        if (cloud_filtered->points.empty()) return;

        pcl::PointCloud<PointT>::Ptr non_ground_cloud = (ground_removal_method_ == "ransac") ? 
            remove_ground_by_ransac(cloud_filtered) : remove_ground_by_ransac(cloud_filtered);

        if (non_ground_cloud->points.empty()) return;

        cluster_and_publish(non_ground_cloud, msg);
    }
    
    void cluster_and_publish(const pcl::PointCloud<PointT>::Ptr& non_ground_cloud, const sensor_msgs::msg::PointCloud::SharedPtr& msg)
    {
        std::vector<pcl::PointIndices> cluster_indices = open3d_dbscan_cluster(non_ground_cloud, dbscan_eps_, dbscan_min_points_);
        
        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::Marker cleanup_marker;
        cleanup_marker.header.frame_id = msg->header.frame_id;
        cleanup_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(cleanup_marker);

        int cluster_id = 0;
        for (const auto& indices : cluster_indices)
        {
            if (indices.indices.empty()) continue;

            // --- MAJOR MODIFICATION: FEATURE ENGINEERING AND INFERENCE ---
            // Step 1: Find min/max intensity for normalization
            float min_intensity = std::numeric_limits<float>::max();
            float max_intensity = std::numeric_limits<float>::lowest();
            for (int index : indices.indices) {
                const auto& point = non_ground_cloud->points[index];
                if (point.intensity < min_intensity) min_intensity = point.intensity;
                if (point.intensity > max_intensity) max_intensity = point.intensity;
            }
            float intensity_range = (max_intensity > min_intensity) ? (max_intensity - min_intensity) : 1.0f;

            // Step 2: Place normalized intensities into z-bins
            std::vector<std::vector<float>> bins(NUM_BINS);
            for (int index : indices.indices) {
                const auto& point = non_ground_cloud->points[index];
                if (point.z >= Z_MIN && point.z < Z_MAX) {
                    int bin_index = static_cast<int>((point.z - Z_MIN) / BIN_WIDTH);
                    float norm_intensity = (point.intensity - min_intensity) / intensity_range;
                    bins[bin_index].push_back(norm_intensity);
                }
            }

            // Step 3: Calculate final feature vector (average of bins)
            std::vector<float> features(NUM_BINS, -1.0f);
            for (int i = 0; i < NUM_BINS; ++i) {
                if (!bins[i].empty()) {
                    features[i] = std::accumulate(bins[i].begin(), bins[i].end(), 0.0f) / bins[i].size();
                }
            }
            
            // Step 4: Run inference to get the classification
            int color_marker_value = predict_cone_type(features);

            // --- Visualization logic (uses model prediction) ---
            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(*non_ground_cloud, indices, centroid);
            
            Eigen::Vector4f min_pt_eigen, max_pt_eigen;
            pcl::getMinMax3D(*non_ground_cloud, indices, min_pt_eigen, max_pt_eigen);

            visualization_msgs::msg::Marker cylinder_marker;
            cylinder_marker.header.frame_id = msg->header.frame_id;
            cylinder_marker.header.stamp = this->get_clock()->now();
            cylinder_marker.ns = "cluster_cylinders";
            cylinder_marker.id = cluster_id;
            cylinder_marker.type = visualization_msgs::msg::Marker::CYLINDER;
            cylinder_marker.action = visualization_msgs::msg::Marker::ADD;
            cylinder_marker.pose.position.x = centroid[0];
            cylinder_marker.pose.position.y = centroid[1];
            cylinder_marker.pose.position.z = (min_pt_eigen[2] + max_pt_eigen[2]) / 2.0;
            cylinder_marker.pose.orientation.w = 1.0;
            cylinder_marker.scale.x = std::max(0.1, (double)max_pt_eigen[0] - min_pt_eigen[0]);
            cylinder_marker.scale.y = std::max(0.1, (double)max_pt_eigen[1] - min_pt_eigen[1]);
            cylinder_marker.scale.z = std::max(0.1, (double)max_pt_eigen[2] - min_pt_eigen[2]);
            
            cylinder_marker.color.a = 1.0;
            if (color_marker_value == 1) { // Blue
                cylinder_marker.color.r = 0.0; cylinder_marker.color.g = 0.0; cylinder_marker.color.b = 1.0;
            } else { // Yellow
                cylinder_marker.color.r = 1.0; cylinder_marker.color.g = 1.0; cylinder_marker.color.b = 0.0;
            }
            cylinder_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
            marker_array.markers.push_back(cylinder_marker);

            // Create text marker
            visualization_msgs::msg::Marker text_marker = cylinder_marker;
            text_marker.ns = "cluster_labels";
            text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text_marker.pose.position.z = max_pt_eigen[2] + 0.3;
            text_marker.text = "Type: " + std::to_string(color_marker_value);
            text_marker.scale.z = 0.4;
            text_marker.color.r = 1.0; text_marker.color.g = 1.0; text_marker.color.b = 1.0; text_marker.color.a = 1.0;
            marker_array.markers.push_back(text_marker);

            cluster_id++;
        }

        if (!marker_array.markers.empty()) {
            marker_publisher_->publish(marker_array);
        }
    }

    // --- Binning constants (must match Python script) ---
    static constexpr int NUM_BINS = 20;
    static constexpr float Z_MIN = -0.110;
    static constexpr float Z_MAX = 0.180;
    static constexpr float BIN_WIDTH = (Z_MAX - Z_MIN) / NUM_BINS;

    // Member variables
    rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr subscription_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
    
    std::thread worker_thread_;
    std::queue<sensor_msgs::msg::PointCloud::SharedPtr> cloud_queue_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> shutdown_flag_;

    long frame_id_ = 0;
    size_t max_queue_size_;

    // Parameters
    std::string ground_removal_method_;
    double y_filter_min_, y_filter_max_, x_filter_min_, x_filter_max_;
    double dbscan_eps_;
    int dbscan_min_points_;
    double ground_height_threshold_;
    double ransac_distance_threshold_;
    int ransac_max_iterations_;
    double ransac_min_z_normal_;
    double ransac_max_slope_deviation_;
    std::string onnx_model_path_;

    // --- NEW: ONNX Runtime members ---
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::vector<const char*> input_node_names_;
    std::vector<const char*> output_node_names_;
    std::string input_name_str_;
    std::string output_name_str_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DataExtractorWithMarkers>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
