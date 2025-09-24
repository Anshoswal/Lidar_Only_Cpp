// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <string>
// #include <thread>
// #include <queue>
// #include <mutex>
// #include <condition_variable>
// #include <atomic>
// #include <algorithm>
// #include <numeric>
// #include <optional> // Required for iterative RANSAC

// #include "rclcpp/rclcpp.hpp"
// #include "sensor_msgs/msg/point_cloud.hpp"
// #include "visualization_msgs/msg/marker_array.hpp"
// #include "visualization_msgs/msg/marker.hpp"

// // PCL specific includes
// #include <pcl_conversions/pcl_conversions.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl/ModelCoefficients.h>
// #include <pcl/segmentation/sac_segmentation.h>
// #include <pcl/filters/extract_indices.h>
// #include <pcl/filters/passthrough.h>
// #include <pcl/common/common.h>
// #include <pcl/common/centroid.h>

// // Open3D for DBSCAN
// #include <open3d/Open3D.h>

// using PointT = pcl::PointXYZI;

// class DataExtractorWithMarkers : public rclcpp::Node
// {
// public:
//     DataExtractorWithMarkers() : Node("data_extractor_with_markers")
//     {
//         // General parameters
//         this->declare_parameter<std::string>("ground_removal_method", "ransac"); // "ransac", "threshold"
//         this->declare_parameter<double>("max_queue_size", 10.0);

//         // Filter parameters
//         this->declare_parameter<double>("y_filter_min", -4.0);
//         this->declare_parameter<double>("y_filter_max", 4.0);
//         this->declare_parameter<double>("x_filter_min", 0.0);
//         this->declare_parameter<double>("x_filter_max", 10.0);

//         // Clustering parameters
//         this->declare_parameter<double>("dbscan_eps", 0.3);
//         this->declare_parameter<int>("dbscan_min_points", 5);

//         // Ground removal parameters
//         this->declare_parameter<double>("ground_height_threshold", -0.1620); // For "threshold" method
//         this->declare_parameter<double>("ransac_distance_threshold", 0.05);  // For "ransac" method
        
//         // --- NEW PARAMETERS FOR ITERATIVE RANSAC ---
//         this->declare_parameter<int>("ransac_max_iterations", 2);
//         this->declare_parameter<double>("ransac_min_z_normal", 0.90);
//         this->declare_parameter<double>("ransac_max_slope_deviation", 10.0);


//         // Get all parameters
//         this->get_parameter("ground_removal_method", ground_removal_method_);
//         this->get_parameter("y_filter_min", y_filter_min_);
//         this->get_parameter("y_filter_max", y_filter_max_);
//         this->get_parameter("x_filter_min", x_filter_min_);
//         this->get_parameter("x_filter_max", x_filter_max_);
//         this->get_parameter("dbscan_eps", dbscan_eps_);
//         this->get_parameter("dbscan_min_points", dbscan_min_points_);
//         this->get_parameter("ground_height_threshold", ground_height_threshold_);
//         this->get_parameter("ransac_distance_threshold", ransac_distance_threshold_);
//         this->get_parameter("ransac_max_iterations", ransac_max_iterations_);
//         this->get_parameter("ransac_min_z_normal", ransac_min_z_normal_);
//         this->get_parameter("ransac_max_slope_deviation", ransac_max_slope_deviation_);
        
//         double max_queue_size;
//         this->get_parameter("max_queue_size", max_queue_size);
//         max_queue_size_ = static_cast<size_t>(max_queue_size);

//         subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud>(
//             "/carmaker/pointcloud",
//             rclcpp::SensorDataQoS(),
//             std::bind(&DataExtractorWithMarkers::lidar_callback, this, std::placeholders::_1));

//         marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/cluster_markers", 10);

//         csv_file_.open("point_level_data.csv");
//         if (!csv_file_.is_open()) {
//             RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file!");
//         } else {
//             // --- CHANGE 1.1: ADD NEW COLUMN TO CSV HEADER ---
//             csv_file_ << "cluster_id,frame_id,z_coordinate,intensity,color_marker\n";
//             csv_file_.flush();
//         }
        
//         RCLCPP_INFO(this->get_logger(), "Data extractor node started.");
//         RCLCPP_INFO(this->get_logger(), "Ground removal method: %s", ground_removal_method_.c_str());
//         if (ground_removal_method_ == "ransac") {
//             RCLCPP_INFO(this->get_logger(), "Iterative RANSAC params: dist_thresh=%.2f, max_iter=%d, min_z_norm=%.2f, max_slope_dev=%.1f",
//                 ransac_distance_threshold_, ransac_max_iterations_, ransac_min_z_normal_, ransac_max_slope_deviation_);
//         } else {
//             RCLCPP_INFO(this->get_logger(), "Height Threshold ground removal: z > %.4f", ground_height_threshold_);
//         }
        
//         shutdown_flag_.store(false);
//         worker_thread_ = std::thread(&DataExtractorWithMarkers::processing_worker, this);
//     }

//     ~DataExtractorWithMarkers()
//     {
//         RCLCPP_INFO(this->get_logger(), "Initiating shutdown...");
//         shutdown_flag_.store(true);
//         condition_.notify_all();
        
//         if (worker_thread_.joinable()) {
//             worker_thread_.join();
//         }

//         RCLCPP_INFO(this->get_logger(), "Writing %zu buffered rows to CSV file...", csv_buffer_.size());
//         {
//             std::lock_guard<std::mutex> csv_lock(csv_mutex_);
//             if (!csv_buffer_.empty()) {
//                 for (const auto& row : csv_buffer_) {
//                     csv_file_ << row << std::endl;
//                 }
//                 csv_buffer_.clear();
//             }
//         }
        
//         if (csv_file_.is_open()) {
//             csv_file_.close();
//             RCLCPP_INFO(this->get_logger(), "CSV file closed.");
//         }
//         RCLCPP_INFO(this->get_logger(), "Shutdown complete.");
//     }
// private:
//     void lidar_callback(const sensor_msgs::msg::PointCloud::SharedPtr msg)
//     {
//         std::lock_guard<std::mutex> lock(queue_mutex_);
//         if (cloud_queue_.size() >= max_queue_size_) {
//             cloud_queue_.pop();
//         }
//         cloud_queue_.push(msg);
//         condition_.notify_one();
//     }

//     void processing_worker()
//     {
//         while (!shutdown_flag_.load()) {
//             sensor_msgs::msg::PointCloud::SharedPtr msg;
//             {
//                 std::unique_lock<std::mutex> lock(queue_mutex_);
//                 condition_.wait(lock, [this] { 
//                     return !cloud_queue_.empty() || shutdown_flag_.load(); 
//                 });

//                 if (shutdown_flag_.load()) break;
                
//                 msg = cloud_queue_.front();
//                 cloud_queue_.pop();
//             }
//             if (msg) {
//                 try {
//                     process_cloud(msg);
//                 } catch (const std::exception& e) {
//                     RCLCPP_ERROR(this->get_logger(), "Exception in process_cloud: %s", e.what());
//                 }
//             }
//         }
//     }

//     std::vector<pcl::PointIndices> open3d_dbscan_cluster(const typename pcl::PointCloud<PointT>::Ptr& cloud, double eps, int min_pts)
//     {
//         if (cloud->points.empty()) return {};
//         auto o3d_pcd = std::make_shared<open3d::geometry::PointCloud>();
//         o3d_pcd->points_.reserve(cloud->points.size());
//         for (const auto& point : cloud->points) {
//             o3d_pcd->points_.emplace_back(point.x, point.y, point.z);
//         }
//         std::vector<int> labels = o3d_pcd->ClusterDBSCAN(eps, min_pts, false);
//         if (labels.empty()) return {};
//         int max_label = *std::max_element(labels.begin(), labels.end());
//         std::vector<pcl::PointIndices> cluster_indices;
//         if (max_label >= 0) {
//             cluster_indices.resize(max_label + 1);
//             for (size_t i = 0; i < labels.size(); ++i) {
//                 if (labels[i] != -1) {
//                     cluster_indices[labels[i]].indices.push_back(i);
//                 }
//             }
//         }
//         return cluster_indices;
//     }

//     pcl::PointCloud<PointT>::Ptr remove_ground_by_height(const pcl::PointCloud<PointT>::Ptr& cloud)
//     {
//         pcl::PointCloud<PointT>::Ptr non_ground_cloud(new pcl::PointCloud<PointT>);
//         pcl::PassThrough<PointT> pass;
//         pass.setInputCloud(cloud);
//         pass.setFilterFieldName("z");
//         pass.setFilterLimits(ground_height_threshold_, 10.0); // keep points > threshold
//         pass.filter(*non_ground_cloud);
//         return non_ground_cloud;
//     }

//     pcl::PointCloud<PointT>::Ptr remove_ground_by_ransac(const pcl::PointCloud<PointT>::Ptr& cloud)
//     {
//         pcl::SACSegmentation<PointT> seg;
//         pcl::ExtractIndices<PointT> extract;
//         pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
//         pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

//         seg.setOptimizeCoefficients(true);
//         seg.setModelType(pcl::SACMODEL_PLANE);
//         seg.setMethodType(pcl::SAC_RANSAC);
//         seg.setDistanceThreshold(ransac_distance_threshold_);

//         auto remaining_cloud = pcl::make_shared<pcl::PointCloud<PointT>>(*cloud);
//         std::optional<Eigen::Vector3f> reference_normal;
//         int iterations = 0;
//         const size_t min_points_for_plane = 100;

//         while (remaining_cloud->points.size() > min_points_for_plane && iterations < ransac_max_iterations_)
//         {
//             seg.setInputCloud(remaining_cloud);
//             seg.segment(*inliers, *coefficients);

//             if (inliers->indices.empty()) {
//                 break;
//             }

//             Eigen::Vector3f current_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
//             if (current_normal.z() < 0) {
//                 current_normal = -current_normal;
//             }

//             if (current_normal.z() < ransac_min_z_normal_) {
//                 break;
//             }

//             if (!reference_normal.has_value()) {
//                 reference_normal = current_normal;
//             } else {
//                 double dot_product = current_normal.dot(reference_normal.value());
//                 double angle_rad = std::acos(std::clamp(static_cast<float>(dot_product), -1.0f, 1.0f));
//                 double angle_deg = angle_rad * (180.0 / M_PI);
//                 if (angle_deg > ransac_max_slope_deviation_) {
//                     break;
//                 }
//             }

//             extract.setInputCloud(remaining_cloud);
//             extract.setIndices(inliers);
//             extract.setNegative(true);
            
//             auto next_remaining_cloud = pcl::make_shared<pcl::PointCloud<PointT>>();
//             extract.filter(*next_remaining_cloud);
//             remaining_cloud = next_remaining_cloud;

//             iterations++;
//         }
        
//         return remaining_cloud;
//     }

//     void process_cloud(const sensor_msgs::msg::PointCloud::SharedPtr msg)
//     {
//         frame_id_++;
//         if (msg->points.empty()) return;

//         pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
//         cloud->points.reserve(msg->points.size());
//         for (size_t i = 0; i < msg->points.size(); ++i) {
//             PointT point;
//             point.x = msg->points[i].x;
//             point.y = msg->points[i].y;
//             point.z = msg->points[i].z;
//             point.intensity = (!msg->channels.empty() && i < msg->channels[0].values.size()) ? msg->channels[0].values[i] : 0.0f;
//             cloud->points.push_back(point);
//         }
//         cloud->width = cloud->points.size();
//         cloud->height = 1;
//         cloud->is_dense = true;

//         pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
//         pcl::PassThrough<PointT> pass;
//         pass.setInputCloud(cloud);
//         pass.setFilterFieldName("y");
//         pass.setFilterLimits(y_filter_min_, y_filter_max_);
//         pass.filter(*cloud_filtered);
//         pass.setInputCloud(cloud_filtered);
//         pass.setFilterFieldName("x");
//         pass.setFilterLimits(x_filter_min_, x_filter_max_);
//         pass.filter(*cloud_filtered);

//         if (cloud_filtered->points.empty()) return;

//         pcl::PointCloud<PointT>::Ptr non_ground_cloud;
//         if (ground_removal_method_ == "threshold") {
//             non_ground_cloud = remove_ground_by_height(cloud_filtered);
//         } else if (ground_removal_method_ == "ransac") {
//             non_ground_cloud = remove_ground_by_ransac(cloud_filtered);
//         } else {
//             RCLCPP_ERROR_ONCE(this->get_logger(), "Unknown ground removal method: %s. Defaulting to ransac.", ground_removal_method_.c_str());
//             non_ground_cloud = remove_ground_by_ransac(cloud_filtered);
//         }

//         if (non_ground_cloud->points.empty()) return;

//         cluster_and_publish(non_ground_cloud, msg);
//     }
    
//     void cluster_and_publish(const pcl::PointCloud<PointT>::Ptr& non_ground_cloud, 
//                             const sensor_msgs::msg::PointCloud::SharedPtr& msg)
//     {
//         std::vector<pcl::PointIndices> cluster_indices = open3d_dbscan_cluster(
//             non_ground_cloud, dbscan_eps_, dbscan_min_points_);
        
//         visualization_msgs::msg::MarkerArray marker_array;
        
//         // Marker to clean up previous markers
//         visualization_msgs::msg::Marker cleanup_marker;
//         cleanup_marker.header.frame_id = msg->header.frame_id;
//         cleanup_marker.action = visualization_msgs::msg::Marker::DELETEALL;
//         marker_array.markers.push_back(cleanup_marker);

//         // --- NEW: FRAME ID MARKER ---
//         // Add a text marker to display the current frame ID
//         visualization_msgs::msg::Marker frame_id_marker;
//         frame_id_marker.header.frame_id = msg->header.frame_id;
//         frame_id_marker.header.stamp = this->get_clock()->now();
//         frame_id_marker.ns = "frame_info";
//         frame_id_marker.id = -1; // Use a unique ID, e.g., -1
//         frame_id_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
//         frame_id_marker.action = visualization_msgs::msg::Marker::ADD;
//         frame_id_marker.pose.position.x = 0; // Position it at the origin or another convenient spot
//         frame_id_marker.pose.position.y = 0;
//         frame_id_marker.pose.position.z = 5.0; // Place it high up to be visible
//         frame_id_marker.scale.z = 0.8; // Text height
//         frame_id_marker.color.a = 1.0;
//         frame_id_marker.color.r = 0.0;
//         frame_id_marker.color.g = 1.0;
//         frame_id_marker.color.b = 0.0;
//         frame_id_marker.text = "Frame: " + std::to_string(frame_id_);
//         frame_id_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
//         marker_array.markers.push_back(frame_id_marker);


//         int cluster_id = 0;
//         for (const auto& indices : cluster_indices)
//         {
//             if (indices.indices.empty()) continue;
            
//             // --- CHANGE 1.2: CALCULATE AVERAGE INTENSITY & DETERMINE COLOR MARKER FIRST ---
//             double total_intensity = 0.0;
//             for (int index : indices.indices) {
//                 total_intensity += non_ground_cloud->points[index].intensity;
//             }
//             double average_intensity = total_intensity / indices.indices.size();

//             // Determine the marker value based on the color condition
//             int color_marker = -1; // Default to -1 (for Yellow)
//             if (average_intensity > 1e6) {
//                 color_marker = 1; // Set to 1 (for Blue)
//             }

//             // --- CHANGE 1.3: LOG EVERY POINT WITH THE NEW COLOR MARKER ---
//             for (int index : indices.indices) {
//                 const auto& point = non_ground_cloud->points[index];
//                 std::stringstream ss;
//                 ss << cluster_id << "," << frame_id_ << "," << point.z << "," << point.intensity << "," << color_marker;
//                 csv_buffer_.push_back(ss.str());
//             }
            
//             // Add a blank line to the CSV buffer after each cluster
//             csv_buffer_.push_back("");

//             // --- Visualization logic ---
//             size_t num_points = indices.indices.size();
//             Eigen::Vector4f centroid;
//             pcl::compute3DCentroid(*non_ground_cloud, indices, centroid);
            
//             Eigen::Vector4f min_pt_eigen, max_pt_eigen;
//             pcl::getMinMax3D(*non_ground_cloud, indices, min_pt_eigen, max_pt_eigen);
//             double bbox_height = max_pt_eigen[2] - min_pt_eigen[2];

//             // Create cylinder marker
//             visualization_msgs::msg::Marker cylinder_marker;
//             cylinder_marker.header.frame_id = msg->header.frame_id;
//             cylinder_marker.header.stamp = this->get_clock()->now();
//             cylinder_marker.ns = "cluster_cylinders";
//             cylinder_marker.id = cluster_id;
//             cylinder_marker.type = visualization_msgs::msg::Marker::CYLINDER;
//             cylinder_marker.action = visualization_msgs::msg::Marker::ADD;
//             cylinder_marker.pose.position.x = centroid[0];
//             cylinder_marker.pose.position.y = centroid[1];
//             cylinder_marker.pose.position.z = (min_pt_eigen[2] + max_pt_eigen[2]) / 2.0;
//             cylinder_marker.pose.orientation.w = 1.0;
//             cylinder_marker.scale.x = std::max(0.1, (double)max_pt_eigen[0] - min_pt_eigen[0]);
//             cylinder_marker.scale.y = std::max(0.1, (double)max_pt_eigen[1] - min_pt_eigen[1]);
//             cylinder_marker.scale.z = std::max(0.1, bbox_height);
            
//             // Set color based on intensity (same logic as before)
//             cylinder_marker.color.a = 1.0;
//             if (average_intensity > 1e6) {
//                 // High intensity = Blue
//                 cylinder_marker.color.r = 0.0;
//                 cylinder_marker.color.g = 0.0;
//                 cylinder_marker.color.b = 1.0;
//             } else {
//                 // Low intensity = Yellow
//                 cylinder_marker.color.r = 1.0;
//                 cylinder_marker.color.g = 1.0;
//                 cylinder_marker.color.b = 0.0;
//             }

//             cylinder_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
//             marker_array.markers.push_back(cylinder_marker);

//             // Create text marker
//             visualization_msgs::msg::Marker text_marker = cylinder_marker;
//             text_marker.ns = "cluster_labels";
//             text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
//             text_marker.pose.position.z = max_pt_eigen[2] + 0.3;
//             // --- CHANGE 2: UPDATE TEXT TO REMOVE INTENSITY ---
//             text_marker.text = "C:" + std::to_string(cluster_id) + "\nP:" + std::to_string(num_points);
//             text_marker.scale.z = 0.4;
//             marker_array.markers.push_back(text_marker);

//             cluster_id++;
//         }

//         // Publish markers if any were generated
//         if (!marker_array.markers.empty()) {
//             marker_publisher_->publish(marker_array);
//         }

//         // Write all buffered data for this frame to the file at once
//         {
//             std::lock_guard<std::mutex> csv_lock(csv_mutex_);
//             if (!csv_buffer_.empty()) {
//                 for (const auto& row : csv_buffer_) {
//                     csv_file_ << row << std::endl;
//                 }
//                 csv_buffer_.clear();
//                 csv_file_.flush(); // Ensure data is written to disk immediately
//             }
//         }
//     }

//     // Member variables
//     rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr subscription_;
//     rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
    
//     std::thread worker_thread_;
//     std::queue<sensor_msgs::msg::PointCloud::SharedPtr> cloud_queue_;
//     std::mutex queue_mutex_;
//     std::condition_variable condition_;
//     std::atomic<bool> shutdown_flag_;

//     std::ofstream csv_file_;
//     std::vector<std::string> csv_buffer_;
//     std::mutex csv_mutex_;
//     long frame_id_ = 0;
//     size_t max_queue_size_;

//     // Parameters
//     std::string ground_removal_method_;
//     double y_filter_min_, y_filter_max_, x_filter_min_, x_filter_max_;
//     double dbscan_eps_;
//     int dbscan_min_points_;
//     double ground_height_threshold_;
//     double ransac_distance_threshold_;
//     int ransac_max_iterations_;
//     double ransac_min_z_normal_;
//     double ransac_max_slope_deviation_;
// };

// int main(int argc, char *argv[])
// {
//     rclcpp::init(argc, argv);
//     auto node = std::make_shared<DataExtractorWithMarkers>();
//     rclcpp::spin(node);
//     rclcpp::shutdown();
//     return 0;
// }

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
#include <sstream>

// ROS2 Includes
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"

// PCL Includes
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

// --- NEW: ONNX Runtime for model inference ---
#include <onnxruntime_cxx_api.h>

using PointT = pcl::PointXYZI;

// --- Binning Configuration (Must match Python script) ---
const int NUM_BINS = 20;
const float Z_MIN = -0.110f;
const float Z_MAX = 0.180f;
const float BIN_WIDTH = (Z_MAX - Z_MIN) / NUM_BINS;


class DataExtractorWithMarkers : public rclcpp::Node
{
public:
    DataExtractorWithMarkers() : Node("data_extractor_with_markers"),
                                 ort_env_(ORT_LOGGING_LEVEL_WARNING, "lidar_classifier")
    {
        // --- Get Parameters ---
        this->declare_parameter<std::string>("ground_removal_method", "ransac");
        this->get_parameter("ground_removal_method", ground_removal_method_);
        // ... (omitting other parameter declarations for brevity) ...
        this->declare_parameter<int>("ransac_max_iterations", 2);
        this->get_parameter("ransac_max_iterations", ransac_max_iterations_);

        // --- NEW: Load the ONNX model and scaler parameters ---
        this->declare_parameter<std::string>("onnx_model_path", "model.onnx");
        this->declare_parameter<std::string>("scaler_params_path", "scaler_params.txt");
        std::string onnx_model_path, scaler_params_path;
        this->get_parameter("onnx_model_path", onnx_model_path);
        this->get_parameter("scaler_params_path", scaler_params_path);

        load_scaler_parameters(scaler_params_path);
        initialize_onnx_session(onnx_model_path);

        // --- Subscriptions, Publishers, CSV setup ---
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud>(
            "/carmaker/pointcloud", rclcpp::SensorDataQoS(),
            std::bind(&DataExtractorWithMarkers::lidar_callback, this, std::placeholders::_1));

        marker_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/cluster_markers", 10);
        
        // Note: The CSV file will now log the PREDICTED marker
        csv_file_.open("point_level_data_predicted.csv");
        if (!csv_file_.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file!");
        } else {
            csv_file_ << "cluster_id,frame_id,z_coordinate,intensity,predicted_color_marker\n";
            csv_file_.flush();
        }

        RCLCPP_INFO(this->get_logger(), "Data extractor node started with ANN classification enabled.");
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
        if (csv_file_.is_open()) {
            csv_file_.close();
        }
        RCLCPP_INFO(this->get_logger(), "Shutdown complete.");
    }

private:
    // --- LIDAR Processing and Clustering (mostly unchanged) ---
    void lidar_callback(const sensor_msgs::msg::PointCloud::SharedPtr msg) { /* ... same as before ... */ }
    void processing_worker() { /* ... same as before ... */ }
    void process_cloud(const sensor_msgs::msg::PointCloud::SharedPtr msg) { /* ... same as before ... */ }
    std::vector<pcl::PointIndices> open3d_dbscan_cluster(const typename pcl::PointCloud<PointT>::Ptr& cloud, double eps, int min_pts) { /* ... same as before ... */ }
    pcl::PointCloud<PointT>::Ptr remove_ground_by_height(const pcl::PointCloud<PointT>::Ptr& cloud) { /* ... same as before ... */ }
    pcl::PointCloud<PointT>::Ptr remove_ground_by_ransac(const pcl::PointCloud<PointT>::Ptr& cloud) { /* ... same as before ... */ }

    // --- NEW: Functions for model loading and inference ---

    void load_scaler_parameters(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            RCLCPP_FATAL(this->get_logger(), "Failed to open scaler params file: %s", path.c_str());
            rclcpp::shutdown();
            return;
        }
        std::string line;
        std::getline(file, line); // Skip header 1
        std::getline(file, line); // Skip header 2

        // Read mean values
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;
            while (std::getline(ss, value, ',')) scaler_mean_.push_back(std::stof(value));
        }
        // Read scale values
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string value;
            while (std::getline(ss, value, ',')) scaler_scale_.push_back(std::stof(value));
        }

        if (scaler_mean_.size() != NUM_BINS || scaler_scale_.size() != NUM_BINS) {
             RCLCPP_FATAL(this->get_logger(), "Scaler params file is malformed. Expected %d values for mean/scale.", NUM_BINS);
             rclcpp::shutdown();
        } else {
             RCLCPP_INFO(this->get_logger(), "Successfully loaded %zu mean and %zu scale values for scaler.", scaler_mean_.size(), scaler_scale_.size());
        }
    }

    void initialize_onnx_session(const std::string& model_path) {
        try {
            ort_session_ = std::make_unique<Ort::Session>(ort_env_, model_path.c_str(), Ort::SessionOptions{nullptr});
            RCLCPP_INFO(this->get_logger(), "ONNX model loaded successfully from: %s", model_path.c_str());
        } catch (const Ort::Exception& e) {
            RCLCPP_FATAL(this->get_logger(), "Failed to load ONNX model: %s", e.what());
            rclcpp::shutdown();
        }
    }

    int predict_with_model(std::vector<float>& feature_vector) {
        if (!ort_session_ || feature_vector.size() != NUM_BINS || scaler_mean_.size() != NUM_BINS) {
            return -1; // Default to yellow if model not loaded or input is wrong size
        }

        // Apply scaling (the same as StandardScaler in Python)
        for (size_t i = 0; i < feature_vector.size(); ++i) {
            feature_vector[i] = (feature_vector[i] - scaler_mean_[i]) / scaler_scale_[i];
        }
        
        // Prepare tensor for ONNX Runtime
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const int64_t input_shape[] = {1, NUM_BINS};
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, feature_vector.data(), feature_vector.size(), input_shape, 2
        );

        const char* input_names[] = {"input_tensor"};
        const char* output_names[] = {"dense_2"}; // IMPORTANT: This name might change. Check with a tool like Netron.

        // Run inference
        try {
            auto output_tensors = ort_session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            
            // Model outputs a probability. Threshold at 0.5 to classify.
            return (output_data[0] > 0.5f) ? 1 : -1; // 1 for Blue, -1 for Yellow

        } catch (const Ort::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "ONNX inference failed: %s", e.what());
            return -1; // Default to yellow on error
        }
    }


    void cluster_and_publish(const pcl::PointCloud<PointT>::Ptr& non_ground_cloud, 
                            const sensor_msgs::msg::PointCloud::SharedPtr& msg)
    {
        std::vector<pcl::PointIndices> cluster_indices = open3d_dbscan_cluster(non_ground_cloud, 0.3, 5);
        visualization_msgs::msg::MarkerArray marker_array;
        
        // ... (cleanup and frame_id marker setup is unchanged) ...

        int cluster_id_counter = 0;
        for (const auto& indices : cluster_indices)
        {
            if (indices.indices.empty()) continue;
            
            // --- PREPROCESSING & INFERENCE PIPELINE ---
            std::vector<PointT> cluster_points;
            std::vector<float> raw_intensities;
            for (int index : indices.indices) {
                const auto& point = non_ground_cloud->points[index];
                cluster_points.push_back(point);
                raw_intensities.push_back(point.intensity);
            }

            // 1. Normalize Intensities (Min-Max) for the current cluster
            auto min_max_it = std::minmax_element(raw_intensities.begin(), raw_intensities.end());
            float min_intensity = *min_max_it.first;
            float max_intensity = *min_max_it.second;
            float intensity_range = max_intensity - min_intensity;
            
            for (size_t i = 0; i < cluster_points.size(); ++i) {
                if (intensity_range > 1e-6) { // Avoid division by zero
                    cluster_points[i].intensity = (cluster_points[i].intensity - min_intensity) / intensity_range;
                } else {
                    cluster_points[i].intensity = 0.0f; // All intensities are the same
                }
            }

            // 2. Bin by Z-coordinate using normalized intensities
            std::vector<std::vector<float>> bins(NUM_BINS);
            for (const auto& point : cluster_points) {
                if (point.z >= Z_MIN && point.z < Z_MAX) {
                    int bin_index = static_cast<int>((point.z - Z_MIN) / BIN_WIDTH);
                    // Ensure bin_index is within bounds
                    if (bin_index >= 0 && bin_index < NUM_BINS) {
                        bins[bin_index].push_back(point.intensity);
                    }
                }
            }
            
            // 3. Create feature vector (average bins or use -1 for empty bins)
            std::vector<float> feature_vector;
            feature_vector.reserve(NUM_BINS);
            for (const auto& bin_content : bins) {
                if (bin_content.empty()) {
                    feature_vector.push_back(-1.0f);
                } else {
                    feature_vector.push_back(std::accumulate(bin_content.begin(), bin_content.end(), 0.0f) / bin_content.size());
                }
            }
            
            // 4. Predict using the ANN model
            int predicted_marker = predict_with_model(feature_vector);


            // --- UPDATE: Use predicted_marker for CSV logging and Visualization ---
            // Log raw intensity values, but with the predicted marker
            for (size_t i = 0; i < cluster_points.size(); ++i) {
                 const auto& point = cluster_points[i];
                 std::stringstream ss;
                 ss << cluster_id_counter << "," << frame_id_ << "," << point.z << "," << raw_intensities[i] << "," << predicted_marker;
                 csv_buffer_.push_back(ss.str());
            }
            csv_buffer_.push_back("");

            // --- Visualization logic using the prediction ---
            // ... (unchanged) ...
            
            cluster_id_counter++;
        }

        // ... (Publishing and CSV writing is unchanged) ...
    }

    // --- Member Variables ---
    rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr subscription_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
    std::thread worker_thread_;
    std::queue<sensor_msgs::msg::PointCloud::SharedPtr> cloud_queue_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> shutdown_flag_;
    std::ofstream csv_file_;
    std::vector<std::string> csv_buffer_;
    long frame_id_ = 0;
    std::string ground_removal_method_;
    int ransac_max_iterations_;
    // ... other parameters ...

    // --- NEW: ONNX and Scaler Members ---
    Ort::Env ort_env_;
    std::unique_ptr<Ort::Session> ort_session_{nullptr};
    std::vector<float> scaler_mean_;
    std::vector<float> scaler_scale_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DataExtractorWithMarkers>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

