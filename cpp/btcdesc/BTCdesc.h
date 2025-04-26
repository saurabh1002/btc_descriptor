#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <iostream>
#include <pcl/common/io.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>

#include <string>
#include <unordered_map>

#define HASH_P 116101
#define MAX_N 10000000000

typedef struct ConfigSetting
{
  /* for submap process*/
  double cloud_ds_size_ = 0.25;

  /* for binary descriptor*/
  int useful_corner_num_ = 30;
  float plane_merge_normal_thre_;
  float plane_merge_dis_thre_;
  float plane_detection_thre_ = 0.01;
  float voxel_size_ = 1.0;
  int voxel_init_num_ = 10;
  int proj_plane_num_ = 1;
  float proj_image_resolution_ = 0.5;
  float proj_image_high_inc_ = 0.5;
  float proj_dis_min_ = 0;
  float proj_dis_max_ = 5;
  float summary_min_thre_ = 10;
  int line_filter_enable_ = 0;

  /* for triangle descriptor */
  float descriptor_near_num_ = 10;
  float descriptor_min_len_ = 1;
  float descriptor_max_len_ = 10;
  float non_max_suppression_radius_ = 3.0;
  float std_side_resolution_ = 0.2;

  /* for place recognition*/
  int skip_near_num_ = 20;
  int candidate_num_ = 50;
  int sub_frame_num_ = 10;
  float rough_dis_threshold_ = 0.03;
  float similarity_threshold_ = 0.7;
  float icp_threshold_ = 0.5;
  float normal_threshold_ = 0.1;
  float dis_threshold_ = 0.3;

} ConfigSetting;

typedef struct BinaryDescriptor
{
  std::vector<bool> occupy_array_;
  unsigned char summary_;
  Eigen::Vector3d location_;
} BinaryDescriptor;

// 1kb,12.8
typedef struct BTC
{
  Eigen::Vector3d triangle_;
  Eigen::Vector3d angle_;
  Eigen::Vector3d center_;
  unsigned short frame_number_;
  // std::vector<unsigned short> score_frame_;
  // std::vector<Eigen::Matrix3d> position_list_;
  BinaryDescriptor binary_A_;
  BinaryDescriptor binary_B_;
  BinaryDescriptor binary_C_;
} BTC;

typedef struct Plane
{
  pcl::PointXYZINormal p_center_;
  Eigen::Vector3d center_;
  Eigen::Vector3d normal_;
  Eigen::Matrix3d covariance_;
  float radius_ = 0;
  float min_eigen_value_ = 1;
  float d_ = 0;
  int id_ = 0;
  int sub_plane_num_ = 0;
  int points_size_ = 0;
  bool is_plane_ = false;
} Plane;

typedef struct BTCMatchList
{
  std::vector<std::pair<BTC, BTC>> match_list_;
  std::pair<int, int> match_id_;
  int match_frame_;
  double mean_dis_;
} BTCMatchList;

struct M_POINT
{
  float xyz[3];
  float intensity;
  int count = 0;
};

class VOXEL_LOC
{
public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOC &other) const
  {
    return (x == other.x && y == other.y && z == other.z);
  }
};

// Hash value
namespace std
{
  template <>
  struct hash<VOXEL_LOC>
  {
    int64_t operator()(const VOXEL_LOC &s) const
    {
      using std::hash;
      using std::size_t;
      return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
    }
  };
} // namespace std

class BTC_LOC
{
public:
  int64_t x, y, z, a, b, c;

  BTC_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0, int64_t va = 0,
          int64_t vb = 0, int64_t vc = 0)
      : x(vx), y(vy), z(vz), a(va), b(vb), c(vc) {}

  bool operator==(const BTC_LOC &other) const
  {
    return (x == other.x && y == other.y && z == other.z);
    // return (x == other.x && y == other.y && z == other.z && a == other.a &&
    //         b == other.b && c == other.c);
  }
};

namespace std
{
  template <>
  struct hash<BTC_LOC>
  {
    int64_t operator()(const BTC_LOC &s) const
    {
      using std::hash;
      using std::size_t;
      return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
    }
  };
} // namespace std

class OctoTree
{
public:
  ConfigSetting config_setting_;
  std::vector<Eigen::Vector3d> voxel_points_;
  Plane *plane_ptr_;
  int layer_;
  int octo_state_; // 0 is end of tree, 1 is not
  int merge_num_ = 0;
  bool is_project_ = false;
  std::vector<Eigen::Vector3d> project_normal;
  bool is_publish_ = false;
  OctoTree *leaves_[8];
  double voxel_center_[3]; // x, y, z
  float quater_length_;
  bool init_octo_;

  // for plot
  bool is_check_connect_[6];
  bool connect_[6];
  OctoTree *connect_tree_[6];

  OctoTree(const ConfigSetting &config_setting)
      : config_setting_(config_setting)
  {
    voxel_points_.clear();
    octo_state_ = 0;
    layer_ = 0;
    init_octo_ = false;
    for (int i = 0; i < 8; i++)
    {
      leaves_[i] = nullptr;
    }
    // for plot
    for (int i = 0; i < 6; i++)
    {
      is_check_connect_[i] = false;
      connect_[i] = false;
      connect_tree_[i] = nullptr;
    }
    plane_ptr_ = new Plane;
  }
  void init_plane();
  void init_octo_tree();
};

void down_sampling_voxel(std::vector<Eigen::Vector3d> &pl_feat,
                         double voxel_size);

void load_config_setting(std::string &config_file,
                         ConfigSetting &config_setting);

double binary_similarity(const BinaryDescriptor &b1,
                         const BinaryDescriptor &b2);

bool binary_greater_sort(BinaryDescriptor a, BinaryDescriptor b);
bool plane_greater_sort(Plane *plane1, Plane *plane2);

double calc_triangle_dis(
    const std::vector<std::pair<BTC, BTC>> &match_std_list);
double calc_binary_similaity(
    const std::vector<std::pair<BTC, BTC>> &match_std_list);

double calc_overlap(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud1,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud2,
                    double dis_threshold);

double calc_overlap(const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud1,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud2,
                    double dis_threshold, int skip_num);

pcl::PointXYZI vec2point(const Eigen::Vector3d &vec);
Eigen::Vector3d point2vec(const pcl::PointXYZI &pi);

Eigen::Vector3d normal2vec(const pcl::PointXYZINormal &pi);

template <typename T>
Eigen::Vector3d point2vec(const T &pi)
{
  Eigen::Vector3d vec(pi.x, pi.y, pi.z);
  return vec;
}

double time_inc(std::chrono::_V2::system_clock::time_point &t_end,
                std::chrono::_V2::system_clock::time_point &t_begin);

struct PlaneSolver
{
  PlaneSolver(Eigen::Vector3d curr_point_, Eigen::Vector3d curr_normal_,
              Eigen::Vector3d target_point_, Eigen::Vector3d target_normal_)
      : curr_point(curr_point_),
        curr_normal(curr_normal_),
        target_point(target_point_),
        target_normal(target_normal_) {};
  template <typename T>
  bool operator()(const T *q, const T *t, T *residual) const
  {
    Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]};
    Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
    Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()),
                              T(curr_point.z())};
    Eigen::Matrix<T, 3, 1> point_w;
    point_w = q_w_curr * cp + t_w_curr;
    Eigen::Matrix<T, 3, 1> point_target(
        T(target_point.x()), T(target_point.y()), T(target_point.z()));
    Eigen::Matrix<T, 3, 1> norm(T(target_normal.x()), T(target_normal.y()),
                                T(target_normal.z()));
    residual[0] = norm.dot(point_w - point_target);
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d curr_point_,
                                     const Eigen::Vector3d curr_normal_,
                                     Eigen::Vector3d target_point_,
                                     Eigen::Vector3d target_normal_)
  {
    return (
        new ceres::AutoDiffCostFunction<PlaneSolver, 1, 4, 3>(new PlaneSolver(
            curr_point_, curr_normal_, target_point_, target_normal_)));
  }

  Eigen::Vector3d curr_point;
  Eigen::Vector3d curr_normal;
  Eigen::Vector3d target_point;
  Eigen::Vector3d target_normal;
};

class BtcDescManager
{
public:
  BtcDescManager() = default;

  ConfigSetting config_setting_;

  unsigned int current_frame_id_;
  unsigned int keyCloudInd;

  std::vector<int> loop_match_ids_;
  std::vector<double> loop_match_scores_;
  std::vector<Eigen::Vector3d> loop_trs_;
  std::vector<Eigen::Matrix3d> loop_rots_;

  BtcDescManager(ConfigSetting &config_setting)
      : config_setting_(config_setting)
  {
    current_frame_id_ = 0;
  };

  // hash table, save all descriptors
  std::unordered_map<BTC_LOC, std::vector<BTC>> data_base_;

  // save all binary descriptors of key frame
  std::vector<std::vector<BinaryDescriptor>> history_binary_list_;

  // save all planes of key frame, required
  std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> plane_cloud_vec_;

  /*Three main processing functions*/

  int ProcessNewScan(const std::vector<Eigen::Vector3d> &pcl);

  std::tuple<int, double, Eigen::Vector3d, Eigen::Matrix3d> GetClosureDataAtIdx(int idx);
  // generate STDescs from a point cloud
  void GenerateSTDescs(pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                       std::vector<BTC> &btcs_vec);

  // search result <candidate_id, plane icp score>. -1 for no loop
  void SearchLoop(const std::vector<BTC> &btcs_vec);

  // add descriptors to database
  void AddSTDescs(const std::vector<BTC> &btcs_vec);

  // Geometrical optimization by plane-to-plane ico
  void PlaneGeomrtricIcp(
      const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
      const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
      std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform);

private:
  /*Following are sub-processing functions*/

  // voxelization and plane detection
  void init_voxel_map(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                      std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map);

  // acquire planes from voxel_map
  void get_plane(const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                 pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud);

  void get_project_plane(std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map,
                         std::vector<Plane *> &project_plane_list);

  void merge_plane(std::vector<Plane *> &origin_list,
                   std::vector<Plane *> &merge_plane_list);

  // extract corner points from pre-build voxel map and clouds

  void binary_extractor(const std::vector<Plane *> proj_plane_list,
                        const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                        std::vector<BinaryDescriptor> &binary_descriptor_list);

  void extract_binary(const Eigen::Vector3d &project_center,
                      const Eigen::Vector3d &project_normal,
                      const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                      std::vector<BinaryDescriptor> &binary_list);

  // non maximum suppression, to control the number of corners
  void non_maxi_suppression(std::vector<BinaryDescriptor> &binary_list);

  // build BTCs from corner points.
  void generate_btc(const std::vector<BinaryDescriptor> &binary_list,
                    const int &frame_id, std::vector<BTC> &btc_list);

  // Select a specified number of candidate frames according to the number of
  // STDesc rough matches
  void candidate_selector(const std::vector<BTC> &btcs_vec,
                          std::vector<BTCMatchList> &candidate_matcher_vec);

  // Get the best candidate frame by geometry check
  void candidate_verify(
      const BTCMatchList &candidate_matcher, double &verify_score,
      std::pair<Eigen::Vector3d, Eigen::Matrix3d> &relative_pose,
      std::vector<std::pair<BTC, BTC>> &sucess_match_vec);

  // Get the transform between a matched std pair
  void triangle_solver(std::pair<BTC, BTC> &std_pair, Eigen::Vector3d &t,
                       Eigen::Matrix3d &rot);

  // Geometrical verification by plane-to-plane icp threshold
  double plane_geometric_verify(
      const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
      const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
      const std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform);
};
