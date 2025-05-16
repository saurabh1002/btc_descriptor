#include "BTCdesc.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <pcl/common/io.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <Eigen/Core>
#include <algorithm>
#include <execution>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "omp.h"

pcl::PointCloud<pcl::PointXYZI>::Ptr EigenToPCL(const std::vector<Eigen::Vector3d> &pointcloud) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl(
        new pcl::PointCloud<pcl::PointXYZI>(pointcloud.size(), 1));
    std::transform(pointcloud.cbegin(), pointcloud.cend(), pcl->begin(),
                   [&](const Eigen::Vector3d &point_eigen) { return vec2point(point_eigen); });
    return pcl;
}

void down_sampling_voxel(std::vector<Eigen::Vector3d> &pl_feat, const double voxel_size) {
    if (voxel_size < 0.01) {
        return;
    }
    auto Discretize = [&](const Eigen::Vector3d &p) -> VOXEL_LOC {
        auto voxel = (p / voxel_size).array().floor().cast<int64_t>();
        return VOXEL_LOC(voxel.x(), voxel.y(), voxel.z());
    };

    std::unordered_map<VOXEL_LOC, M_POINT> voxel_map;
    std::for_each(pl_feat.cbegin(), pl_feat.cend(), [&](const Eigen::Vector3d &point) {
        VOXEL_LOC position = Discretize(point);
        auto iter = voxel_map.find(position);
        if (iter != voxel_map.end()) {
            iter->second.point += point;
            iter->second.count += 1;
        } else {
            M_POINT anp;
            anp.point = point;
            anp.count = 1;
            voxel_map[position] = anp;
        }
    });

    pl_feat.resize(voxel_map.size());
    std::transform(voxel_map.cbegin(), voxel_map.cend(), pl_feat.begin(),
                   [](const auto &pair) { return pair.second.point / pair.second.count; });
}

auto binary_similarity = [](const BinaryDescriptor &b1, const BinaryDescriptor &b2) -> double {
    double dis = 0;
    for (size_t i = 0; i < b1.occupy_array_.size(); i++) {
        // to be debug hanming distance
        if (b1.occupy_array_[i] == true && b2.occupy_array_[i] == true) {
            dis += 1;
        }
    }
    return 2 * dis / (b1.summary_ + b2.summary_);
};

auto binary_greater_sort = [](const BinaryDescriptor &a, const BinaryDescriptor &b) -> bool {
    return (a.summary_ > b.summary_);
};

auto plane_greater_sort = [](const Plane *plane1, const Plane *plane2) -> bool {
    return plane1->points_size_ > plane2->points_size_;
};

void OctoTree::init_octo_tree() {
    if (voxel_points_.size() > config_setting_.voxel_init_num_) {
        init_plane();
    }
}

void OctoTree::init_plane() {
    plane_ptr_->covariance_ = Eigen::Matrix3d::Zero();
    plane_ptr_->center_ = Eigen::Vector3d::Zero();
    plane_ptr_->normal_ = Eigen::Vector3d::Zero();
    plane_ptr_->points_size_ = voxel_points_.size();
    plane_ptr_->radius_ = 0;
    for (auto pi : voxel_points_) {
        plane_ptr_->covariance_ += pi * pi.transpose();
        plane_ptr_->center_ += pi;
    }
    plane_ptr_->center_ = plane_ptr_->center_ / plane_ptr_->points_size_;
    plane_ptr_->covariance_ = plane_ptr_->covariance_ / plane_ptr_->points_size_ -
                              plane_ptr_->center_ * plane_ptr_->center_.transpose();
    Eigen::EigenSolver<Eigen::Matrix3d> es(plane_ptr_->covariance_);
    const Eigen::Matrix3cd evecs = es.eigenvectors();
    const Eigen::Vector3cd evals = es.eigenvalues();
    const Eigen::Vector3d evalsReal = evals.real();
    Eigen::Matrix3d::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    if (evalsReal(evalsMin) < config_setting_.plane_detection_thre_) {
        plane_ptr_->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
            evecs.real()(2, evalsMin);
        plane_ptr_->min_eigen_value_ = evalsReal(evalsMin);
        plane_ptr_->radius_ = sqrt(evalsReal(evalsMax));
        plane_ptr_->is_plane_ = true;

        plane_ptr_->d_ = -(plane_ptr_->normal_(0) * plane_ptr_->center_(0) +
                           plane_ptr_->normal_(1) * plane_ptr_->center_(1) +
                           plane_ptr_->normal_(2) * plane_ptr_->center_(2));
        plane_ptr_->p_center_.x = plane_ptr_->center_(0);
        plane_ptr_->p_center_.y = plane_ptr_->center_(1);
        plane_ptr_->p_center_.z = plane_ptr_->center_(2);
        plane_ptr_->p_center_.normal_x = plane_ptr_->normal_(0);
        plane_ptr_->p_center_.normal_y = plane_ptr_->normal_(1);
        plane_ptr_->p_center_.normal_z = plane_ptr_->normal_(2);
    } else {
        plane_ptr_->is_plane_ = false;
    }
}

inline pcl::PointXYZI vec2point(const Eigen::Vector3d &vec) {
    pcl::PointXYZI pi;
    pi.x = vec[0];
    pi.y = vec[1];
    pi.z = vec[2];
    return pi;
}

inline Eigen::Vector3d point2vec(const pcl::PointXYZ &pi) {
    return Eigen::Vector3d(pi.x, pi.y, pi.z);
}

inline Eigen::Vector3d point2vec(const pcl::PointXYZI &pi) {
    return Eigen::Vector3d(pi.x, pi.y, pi.z);
}

inline Eigen::Vector3d point2vec(const pcl::PointXYZINormal &pi) {
    return Eigen::Vector3d(pi.x, pi.y, pi.z);
}

inline Eigen::Vector3d normal2vec(const pcl::PointXYZINormal &pi) {
    return Eigen::Vector3d(pi.normal_x, pi.normal_y, pi.normal_z);
}

void BtcDescManager::GenerateBTCDescs(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::vector<BTC> &btcs_vec) {  // step1, voxelization and plane dection
    std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;
    init_voxel_map(input_cloud, voxel_map);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr plane_cloud(
        new pcl::PointCloud<pcl::PointXYZINormal>);
    get_plane(voxel_map, plane_cloud);
    plane_cloud_vec_.emplace_back(plane_cloud);

    // step3, extraction binary descriptors
    std::vector<Plane *> proj_plane_list;
    std::vector<Plane *> merge_plane_list;
    get_project_plane(voxel_map, proj_plane_list);
    if (proj_plane_list.size() == 0) {
        Plane *single_plane = new Plane;
        single_plane->normal_ << 0, 0, 1;
        single_plane->center_ << input_cloud->points[0].x, input_cloud->points[0].y,
            input_cloud->points[0].z;
        merge_plane_list.emplace_back(single_plane);
    } else {
        sort(proj_plane_list.begin(), proj_plane_list.end(), plane_greater_sort);
        merge_plane(proj_plane_list, merge_plane_list);
        sort(merge_plane_list.begin(), merge_plane_list.end(), plane_greater_sort);
    }
    std::vector<BinaryDescriptor> binary_list;
    binary_extractor(merge_plane_list, input_cloud, binary_list);
    history_binary_list_.emplace_back(binary_list);

    // step4, generate stable triangle descriptors
    btcs_vec.clear();
    generate_btc(binary_list, current_frame_id_, btcs_vec);

    // step5, clear memory
    std::for_each(voxel_map.begin(), voxel_map.end(), [](auto &pair) { delete (pair.second); });
}

void BtcDescManager::SearchLoop(const std::vector<BTC> &btcs_vec) {
    if (btcs_vec.size() == 0) {
        std::cout << "No STDescs!\n";
        return;
    }
    // step1, select candidates, default number 50
    std::vector<BTCMatchList> candidate_matcher_vec;
    candidate_selector(btcs_vec, candidate_matcher_vec);

    loop_match_ids_.resize(candidate_matcher_vec.size());
    loop_match_scores_.resize(candidate_matcher_vec.size());
    loop_rots_.resize(candidate_matcher_vec.size());
    loop_trs_.resize(candidate_matcher_vec.size());
    // step2, select best candidates from rough candidates
    for (size_t i = 0; i < candidate_matcher_vec.size(); i++) {
        double verify_score = -1;
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> relative_pose;
        std::vector<std::pair<BTC, BTC>> sucess_match_vec;
        candidate_verify(candidate_matcher_vec[i], verify_score, relative_pose, sucess_match_vec);
        loop_match_ids_[i] = candidate_matcher_vec[i].match_id_.second;
        loop_match_scores_[i] = verify_score;
        loop_rots_[i] = relative_pose.second;
        loop_trs_[i] = relative_pose.first;
    }
}

void BtcDescManager::AddBTCDescs(const std::vector<BTC> &btcs_vec) {
    // update frame id
    current_frame_id_++;
    std::for_each(btcs_vec.cbegin(), btcs_vec.cend(), [&](const auto &single_btc) {
        // calculate the position of single std
        BTC_LOC position;
        position.x = (int)(single_btc.triangle_[0] + 0.5);
        position.y = (int)(single_btc.triangle_[1] + 0.5);
        position.z = (int)(single_btc.triangle_[2] + 0.5);
        auto iter = data_base_.find(position);
        if (iter != data_base_.end()) {
            data_base_[position].emplace_back(single_btc);
        } else {
            data_base_[position] = std::vector<BTC>{single_btc};
        }
    });
}

void BtcDescManager::init_voxel_map(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                                    std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map) {
    auto Discretize = [&](const Eigen::Vector3d &p) -> VOXEL_LOC {
        auto voxel = (p / config_setting_.voxel_size_).array().floor().cast<int64_t>();
        return VOXEL_LOC(voxel.x(), voxel.y(), voxel.z());
    };
    std::for_each(input_cloud->points.cbegin(), input_cloud->points.cend(),
                  [&](const pcl::PointXYZI &point) {
                      Eigen::Vector3d p_c = point2vec(point);
                      VOXEL_LOC position = Discretize(p_c);
                      auto iter = voxel_map.find(position);
                      if (iter != voxel_map.end()) {
                          voxel_map[position]->voxel_points_.emplace_back(p_c);
                      } else {
                          voxel_map[position] = new OctoTree(config_setting_);
                          voxel_map[position]->voxel_points_.emplace_back(p_c);
                      }
                  });

    std::for_each(voxel_map.begin(), voxel_map.end(),
                  [](auto &pair) { pair.second->init_octo_tree(); });
}

void BtcDescManager::get_plane(const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                               pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud) {
    plane_cloud->reserve(voxel_map.size());
    std::for_each(voxel_map.cbegin(), voxel_map.cend(), [&](const auto &pair) {
        if (pair.second->plane_ptr_->is_plane_) {
            pcl::PointXYZINormal pi;
            pi.x = pair.second->plane_ptr_->center_[0];
            pi.y = pair.second->plane_ptr_->center_[1];
            pi.z = pair.second->plane_ptr_->center_[2];
            pi.normal_x = pair.second->plane_ptr_->normal_[0];
            pi.normal_y = pair.second->plane_ptr_->normal_[1];
            pi.normal_z = pair.second->plane_ptr_->normal_[2];
            plane_cloud->emplace_back(pi);
        }
    });
    plane_cloud->points.shrink_to_fit();
}

void BtcDescManager::get_project_plane(std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                                       std::vector<Plane *> &project_plane_list) {
    std::vector<Plane *> origin_list;
    origin_list.reserve(voxel_map.size());
    std::for_each(voxel_map.cbegin(), voxel_map.cend(), [&](const auto &pair) {
        if (pair.second->plane_ptr_->is_plane_) {
            origin_list.emplace_back(pair.second->plane_ptr_);
        }
    });
    origin_list.shrink_to_fit();

    std::for_each(origin_list.begin(), origin_list.end(), [](auto &origin) { origin->id_ = 0; });
    int current_id = 1;
    for (auto iter = origin_list.end() - 1; iter != origin_list.begin(); iter--) {
        for (auto iter2 = origin_list.begin(); iter2 != iter; iter2++) {
            const Eigen::Vector3d normal_diff = (*iter)->normal_ - (*iter2)->normal_;
            const Eigen::Vector3d normal_add = (*iter)->normal_ + (*iter2)->normal_;
            const double dis1 = fabs((*iter)->normal_(0) * (*iter2)->center_(0) +
                                     (*iter)->normal_(1) * (*iter2)->center_(1) +
                                     (*iter)->normal_(2) * (*iter2)->center_(2) + (*iter)->d_);
            const double dis2 = fabs((*iter2)->normal_(0) * (*iter)->center_(0) +
                                     (*iter2)->normal_(1) * (*iter)->center_(1) +
                                     (*iter2)->normal_(2) * (*iter)->center_(2) + (*iter2)->d_);
            if (normal_diff.norm() < config_setting_.plane_merge_normal_thre_ ||
                normal_add.norm() < config_setting_.plane_merge_normal_thre_)
                if (dis1 < config_setting_.plane_merge_dis_thre_ &&
                    dis2 < config_setting_.plane_merge_dis_thre_) {
                    if ((*iter)->id_ == 0 && (*iter2)->id_ == 0) {
                        (*iter)->id_ = current_id;
                        (*iter2)->id_ = current_id;
                        current_id++;
                    } else if ((*iter)->id_ == 0 && (*iter2)->id_ != 0)
                        (*iter)->id_ = (*iter2)->id_;
                    else if ((*iter)->id_ != 0 && (*iter2)->id_ == 0)
                        (*iter2)->id_ = (*iter)->id_;
                }
        }
    }
    std::vector<Plane *> merge_list;
    std::vector<int> merge_flag;
    merge_list.reserve(origin_list.size());
    merge_flag.reserve(origin_list.size());
    for (size_t i = 0; i < origin_list.size(); i++) {
        auto it = std::find(merge_flag.cbegin(), merge_flag.cend(), origin_list[i]->id_);
        if (it != merge_flag.end()) continue;
        if (origin_list[i]->id_ == 0) {
            continue;
        }
        Plane *merge_plane = new Plane;
        (*merge_plane) = (*origin_list[i]);
        bool is_merge = false;
        for (size_t j = 0; j < origin_list.size(); j++) {
            if (i == j) continue;
            if (origin_list[j]->id_ == origin_list[i]->id_) {
                is_merge = true;
                const Eigen::Matrix3d P_PT1 =
                    (merge_plane->covariance_ +
                     merge_plane->center_ * merge_plane->center_.transpose()) *
                    merge_plane->points_size_;
                const Eigen::Matrix3d P_PT2 =
                    (origin_list[j]->covariance_ +
                     origin_list[j]->center_ * origin_list[j]->center_.transpose()) *
                    origin_list[j]->points_size_;
                const Eigen::Vector3d merge_center =
                    (merge_plane->center_ * merge_plane->points_size_ +
                     origin_list[j]->center_ * origin_list[j]->points_size_) /
                    (merge_plane->points_size_ + origin_list[j]->points_size_);
                const Eigen::Matrix3d merge_covariance =
                    (P_PT1 + P_PT2) / (merge_plane->points_size_ + origin_list[j]->points_size_) -
                    merge_center * merge_center.transpose();
                merge_plane->covariance_ = merge_covariance;
                merge_plane->center_ = merge_center;
                merge_plane->points_size_ =
                    merge_plane->points_size_ + origin_list[j]->points_size_;
                merge_plane->sub_plane_num_++;

                Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance_);
                const Eigen::Matrix3cd evecs = es.eigenvectors();
                const Eigen::Vector3cd evals = es.eigenvalues();
                const Eigen::Vector3d evalsReal = evals.real();
                Eigen::Matrix3f::Index evalsMin, evalsMax;
                evalsReal.rowwise().sum().minCoeff(&evalsMin);
                evalsReal.rowwise().sum().maxCoeff(&evalsMax);
                merge_plane->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
                    evecs.real()(2, evalsMin);
                merge_plane->radius_ = sqrt(evalsReal(evalsMax));
                merge_plane->d_ = -(merge_plane->normal_(0) * merge_plane->center_(0) +
                                    merge_plane->normal_(1) * merge_plane->center_(1) +
                                    merge_plane->normal_(2) * merge_plane->center_(2));
                merge_plane->p_center_.x = merge_plane->center_(0);
                merge_plane->p_center_.y = merge_plane->center_(1);
                merge_plane->p_center_.z = merge_plane->center_(2);
                merge_plane->p_center_.normal_x = merge_plane->normal_(0);
                merge_plane->p_center_.normal_y = merge_plane->normal_(1);
                merge_plane->p_center_.normal_z = merge_plane->normal_(2);
            }
        }
        if (is_merge) {
            merge_flag.emplace_back(merge_plane->id_);
            merge_list.emplace_back(merge_plane);
        }
    }
    merge_list.shrink_to_fit();
    merge_flag.shrink_to_fit();
    project_plane_list = merge_list;
}

void BtcDescManager::merge_plane(std::vector<Plane *> &origin_list,
                                 std::vector<Plane *> &merge_plane_list) {
    if (origin_list.size() == 1) {
        merge_plane_list = origin_list;
        return;
    }
    for (size_t i = 0; i < origin_list.size(); i++) origin_list[i]->id_ = 0;
    int current_id = 1;
    for (auto iter = origin_list.end() - 1; iter != origin_list.begin(); iter--) {
        for (auto iter2 = origin_list.begin(); iter2 != iter; iter2++) {
            const Eigen::Vector3d normal_diff = (*iter)->normal_ - (*iter2)->normal_;
            const Eigen::Vector3d normal_add = (*iter)->normal_ + (*iter2)->normal_;
            const double dis1 = fabs((*iter)->normal_(0) * (*iter2)->center_(0) +
                                     (*iter)->normal_(1) * (*iter2)->center_(1) +
                                     (*iter)->normal_(2) * (*iter2)->center_(2) + (*iter)->d_);
            const double dis2 = fabs((*iter2)->normal_(0) * (*iter)->center_(0) +
                                     (*iter2)->normal_(1) * (*iter)->center_(1) +
                                     (*iter2)->normal_(2) * (*iter)->center_(2) + (*iter2)->d_);
            if (normal_diff.norm() < config_setting_.plane_merge_normal_thre_ ||
                normal_add.norm() < config_setting_.plane_merge_normal_thre_)
                if (dis1 < config_setting_.plane_merge_dis_thre_ &&
                    dis2 < config_setting_.plane_merge_dis_thre_) {
                    if ((*iter)->id_ == 0 && (*iter2)->id_ == 0) {
                        (*iter)->id_ = current_id;
                        (*iter2)->id_ = current_id;
                        current_id++;
                    } else if ((*iter)->id_ == 0 && (*iter2)->id_ != 0)
                        (*iter)->id_ = (*iter2)->id_;
                    else if ((*iter)->id_ != 0 && (*iter2)->id_ == 0)
                        (*iter2)->id_ = (*iter)->id_;
                }
        }
    }

    std::vector<int> merge_flag;
    merge_flag.reserve(origin_list.size());
    for (size_t i = 0; i < origin_list.size(); i++) {
        auto it = std::find(merge_flag.cbegin(), merge_flag.cend(), origin_list[i]->id_);
        if (it != merge_flag.end()) continue;
        if (origin_list[i]->id_ == 0) {
            merge_plane_list.emplace_back(origin_list[i]);
            continue;
        }
        Plane *merge_plane = new Plane;
        (*merge_plane) = (*origin_list[i]);
        bool is_merge = false;
        for (size_t j = 0; j < origin_list.size(); j++) {
            if (i == j) continue;
            if (origin_list[j]->id_ == origin_list[i]->id_) {
                is_merge = true;
                const Eigen::Matrix3d P_PT1 =
                    (merge_plane->covariance_ +
                     merge_plane->center_ * merge_plane->center_.transpose()) *
                    merge_plane->points_size_;
                const Eigen::Matrix3d P_PT2 =
                    (origin_list[j]->covariance_ +
                     origin_list[j]->center_ * origin_list[j]->center_.transpose()) *
                    origin_list[j]->points_size_;
                const Eigen::Vector3d merge_center =
                    (merge_plane->center_ * merge_plane->points_size_ +
                     origin_list[j]->center_ * origin_list[j]->points_size_) /
                    (merge_plane->points_size_ + origin_list[j]->points_size_);
                const Eigen::Matrix3d merge_covariance =
                    (P_PT1 + P_PT2) / (merge_plane->points_size_ + origin_list[j]->points_size_) -
                    merge_center * merge_center.transpose();
                merge_plane->covariance_ = merge_covariance;
                merge_plane->center_ = merge_center;
                merge_plane->points_size_ =
                    merge_plane->points_size_ + origin_list[j]->points_size_;
                merge_plane->sub_plane_num_ += origin_list[j]->sub_plane_num_;

                Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance_);
                const Eigen::Matrix3cd evecs = es.eigenvectors();
                const Eigen::Vector3cd evals = es.eigenvalues();
                const Eigen::Vector3d evalsReal = evals.real();
                Eigen::Matrix3f::Index evalsMin, evalsMax;
                evalsReal.rowwise().sum().minCoeff(&evalsMin);
                evalsReal.rowwise().sum().maxCoeff(&evalsMax);
                merge_plane->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
                    evecs.real()(2, evalsMin);
                merge_plane->radius_ = sqrt(evalsReal(evalsMax));
                merge_plane->d_ = -(merge_plane->normal_(0) * merge_plane->center_(0) +
                                    merge_plane->normal_(1) * merge_plane->center_(1) +
                                    merge_plane->normal_(2) * merge_plane->center_(2));
                merge_plane->p_center_.x = merge_plane->center_(0);
                merge_plane->p_center_.y = merge_plane->center_(1);
                merge_plane->p_center_.z = merge_plane->center_(2);
                merge_plane->p_center_.normal_x = merge_plane->normal_(0);
                merge_plane->p_center_.normal_y = merge_plane->normal_(1);
                merge_plane->p_center_.normal_z = merge_plane->normal_(2);
            }
        }
        if (is_merge) {
            merge_flag.emplace_back(merge_plane->id_);
            merge_plane_list.emplace_back(merge_plane);
        }
    }
}

void BtcDescManager::binary_extractor(const std::vector<Plane *> proj_plane_list,
                                      const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                                      std::vector<BinaryDescriptor> &binary_descriptor_list) {
    int useful_proj_num = 0;
    Eigen::Vector3d last_normal(0, 0, 0);
    std::vector<BinaryDescriptor> temp_binary_list;
    for (int i = 0; i < proj_plane_list.size(); i++) {
        std::vector<BinaryDescriptor> prepare_binary_list;
        const Eigen::Vector3d &proj_center = proj_plane_list[i]->center_;
        const Eigen::Vector3d &proj_normal = proj_plane_list[i]->normal_;
        if ((proj_normal - last_normal).norm() < 0.3 || (proj_normal + last_normal).norm() > 0.3) {
            last_normal = proj_normal;
            useful_proj_num++;
            extract_binary(proj_center, proj_normal, input_cloud, prepare_binary_list);
            for (const auto &bi : prepare_binary_list) {
                temp_binary_list.emplace_back(bi);
            }
            if (useful_proj_num == config_setting_.proj_plane_num_) {
                break;
            }
        }
    }
    non_maxi_suppression(temp_binary_list);
    binary_descriptor_list.clear();
    if (config_setting_.useful_corner_num_ > temp_binary_list.size()) {
        binary_descriptor_list = temp_binary_list;
    } else {
        std::sort(temp_binary_list.begin(), temp_binary_list.end(), binary_greater_sort);
        for (size_t i = 0; i < config_setting_.useful_corner_num_; i++) {
            binary_descriptor_list.emplace_back(temp_binary_list[i]);
        }
    }
}

void BtcDescManager::extract_binary(const Eigen::Vector3d &project_center,
                                    const Eigen::Vector3d &project_normal,
                                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                                    std::vector<BinaryDescriptor> &binary_list) {
    binary_list.clear();
    const double binary_min_dis = config_setting_.summary_min_thre_;
    const double resolution = config_setting_.proj_image_resolution_;
    const double dis_threshold_min = config_setting_.proj_dis_min_;
    const double dis_threshold_max = config_setting_.proj_dis_max_;
    const double high_inc = config_setting_.proj_image_high_inc_;
    const bool line_filter_enable = config_setting_.line_filter_enable_;
    const double A = project_normal[0];
    const double B = project_normal[1];
    const double C = project_normal[2];
    const double D = -(A * project_center[0] + B * project_center[1] + C * project_center[2]);

    Eigen::Vector3d x_axis(1, 0, 0);
    if (C != 0) {
        x_axis[2] = -(A + B) / C;
    } else if (B != 0) {
        x_axis[1] = -A / B;
    } else {
        x_axis[0] = 0;
        x_axis[1] = 1;
    }
    x_axis.normalize();

    Eigen::Vector3d y_axis = project_normal.cross(x_axis);
    y_axis.normalize();
    const double ax = x_axis[0];
    const double bx = x_axis[1];
    const double cx = x_axis[2];
    const double dx = -(ax * project_center[0] + bx * project_center[1] + cx * project_center[2]);
    const double ay = y_axis[0];
    const double by = y_axis[1];
    const double cy = y_axis[2];
    const double dy = -(ay * project_center[0] + by * project_center[1] + cy * project_center[2]);

    std::vector<Eigen::Vector2d> point_list_2d;
    pcl::PointCloud<pcl::PointXYZ> point_list_3d;
    std::vector<double> dis_list_2d;
    dis_list_2d.reserve(input_cloud->size());
    point_list_2d.reserve(input_cloud->size());
    point_list_3d.reserve(input_cloud->size());
    for (size_t i = 0; i < input_cloud->size(); i++) {
        const double x = input_cloud->points[i].x;
        const double y = input_cloud->points[i].y;
        const double z = input_cloud->points[i].z;
        const double dis = fabs(x * A + y * B + z * C + D);
        pcl::PointXYZ pi;
        if (dis < dis_threshold_min || dis > dis_threshold_max) {
            continue;
        } else {
            if (dis > dis_threshold_min && dis <= dis_threshold_max) {
                pi.x = x;
                pi.y = y;
                pi.z = z;
            }
        }
        Eigen::Vector3d cur_project;
        cur_project[0] = (-A * (B * y + C * z + D) + x * (B * B + C * C)) / (A * A + B * B + C * C);
        cur_project[1] = (-B * (A * x + C * z + D) + y * (A * A + C * C)) / (A * A + B * B + C * C);
        cur_project[2] = (-C * (A * x + B * y + D) + z * (A * A + B * B)) / (A * A + B * B + C * C);

        const double project_x =
            cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy;
        const double project_y =
            cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx;
        dis_list_2d.emplace_back(dis);
        point_list_2d.emplace_back(project_x, project_y);
        point_list_3d.points.emplace_back(pi);
    }

    double min_x = 10;
    double max_x = -10;
    double min_y = 10;
    double max_y = -10;
    if (point_list_2d.size() <= 5) {
        return;
    }
    for (const auto &pi : point_list_2d) {
        if (pi[0] < min_x) {
            min_x = pi[0];
        }
        if (pi[0] > max_x) {
            max_x = pi[0];
        }
        if (pi[1] < min_y) {
            min_y = pi[1];
        }
        if (pi[1] > max_y) {
            max_y = pi[1];
        }
    }
    // segment project cloud
    const int segmen_base_num = 5;
    const double segmen_len = segmen_base_num * resolution;
    const int x_segment_num = (max_x - min_x) / segmen_len + 1;
    const int y_segment_num = (max_y - min_y) / segmen_len + 1;
    const int x_axis_len = (int)((max_x - min_x) / resolution + segmen_base_num);
    const int y_axis_len = (int)((max_y - min_y) / resolution + segmen_base_num);

    std::vector<double> **dis_container = new std::vector<double> *[x_axis_len];
    BinaryDescriptor **binary_container = new BinaryDescriptor *[x_axis_len];
    for (int i = 0; i < x_axis_len; i++) {
        dis_container[i] = new std::vector<double>[y_axis_len];
        binary_container[i] = new BinaryDescriptor[y_axis_len];
    }
    double **img_count = new double *[x_axis_len];
    for (int i = 0; i < x_axis_len; i++) {
        img_count[i] = new double[y_axis_len];
    }
    double **dis_array = new double *[x_axis_len];
    for (int i = 0; i < x_axis_len; i++) {
        dis_array[i] = new double[y_axis_len];
    }
    double **mean_x_list = new double *[x_axis_len];
    for (int i = 0; i < x_axis_len; i++) {
        mean_x_list[i] = new double[y_axis_len];
    }
    double **mean_y_list = new double *[x_axis_len];
    for (int i = 0; i < x_axis_len; i++) {
        mean_y_list[i] = new double[y_axis_len];
    }
    for (int x = 0; x < x_axis_len; x++) {
        for (int y = 0; y < y_axis_len; y++) {
            img_count[x][y] = 0;
            mean_x_list[x][y] = 0;
            mean_y_list[x][y] = 0;
            dis_array[x][y] = 0;
            dis_container[x][y] = std::vector<double>{};
        }
    }

    for (size_t i = 0; i < point_list_2d.size(); i++) {
        int x_index = (int)((point_list_2d[i][0] - min_x) / resolution);
        int y_index = (int)((point_list_2d[i][1] - min_y) / resolution);
        mean_x_list[x_index][y_index] += point_list_2d[i][0];
        mean_y_list[x_index][y_index] += point_list_2d[i][1];
        img_count[x_index][y_index]++;
        dis_container[x_index][y_index].emplace_back(dis_list_2d[i]);
    }

    for (int x = 0; x < x_axis_len; x++) {
        for (int y = 0; y < y_axis_len; y++) {
            // calc segment dis array
            if (img_count[x][y] > 0) {
                const int cut_num = (dis_threshold_max - dis_threshold_min) / high_inc;
                std::vector<bool> occup_list;
                std::vector<double> cnt_list;
                for (size_t i = 0; i < cut_num; i++) {
                    cnt_list.emplace_back(0);
                    occup_list.emplace_back(false);
                }
                for (size_t j = 0; j < dis_container[x][y].size(); j++) {
                    const int cnt_index = (dis_container[x][y][j] - dis_threshold_min) / high_inc;
                    cnt_list[cnt_index]++;
                }

                double segmnt_dis = 0;
                for (size_t i = 0; i < cut_num; i++) {
                    if (cnt_list[i] >= 1) {
                        segmnt_dis++;
                        occup_list[i] = true;
                    }
                }

                BinaryDescriptor single_binary;
                dis_array[x][y] = segmnt_dis;
                single_binary.occupy_array_ = occup_list;
                single_binary.summary_ = segmnt_dis;
                binary_container[x][y] = single_binary;
            }
        }
    }

    // filter by distance
    std::vector<double> max_dis_list;
    std::vector<int> max_dis_x_index_list;
    std::vector<int> max_dis_y_index_list;
    max_dis_list.reserve(x_segment_num * y_segment_num);
    max_dis_x_index_list.reserve(x_segment_num * y_segment_num);
    max_dis_y_index_list.reserve(x_segment_num * y_segment_num);
    for (int x_segment_index = 0; x_segment_index < x_segment_num; x_segment_index++) {
        for (int y_segment_index = 0; y_segment_index < y_segment_num; y_segment_index++) {
            double max_dis = 0;
            int max_dis_x_index = -10;
            int max_dis_y_index = -10;
            for (int x_index = x_segment_index * segmen_base_num;
                 x_index < (x_segment_index + 1) * segmen_base_num; x_index++) {
                for (int y_index = y_segment_index * segmen_base_num;
                     y_index < (y_segment_index + 1) * segmen_base_num; y_index++) {
                    if (dis_array[x_index][y_index] > max_dis) {
                        max_dis = dis_array[x_index][y_index];
                        max_dis_x_index = x_index;
                        max_dis_y_index = y_index;
                    }
                }
            }
            if (max_dis >= binary_min_dis) {
                max_dis_list.emplace_back(max_dis);
                max_dis_x_index_list.emplace_back(max_dis_x_index);
                max_dis_y_index_list.emplace_back(max_dis_y_index);
            }
        }
    }
    max_dis_list.shrink_to_fit();
    max_dis_x_index_list.shrink_to_fit();
    max_dis_y_index_list.shrink_to_fit();
    // calc line or not
    std::vector<Eigen::Vector2i> direction_list = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (size_t i = 0; i < max_dis_list.size(); i++) {
        Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
        if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 || p[1] >= y_axis_len - 1) {
            continue;
        }
        bool is_add = true;

        if (line_filter_enable) {
            for (int j = 0; j < 4; j++) {
                Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
                if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 || p[1] >= y_axis_len - 1) {
                    continue;
                }
                const Eigen::Vector2i p1 = p + direction_list[j];
                const Eigen::Vector2i p2 = p - direction_list[j];
                double threshold = dis_array[p[0]][p[1]] - 3;
                if (dis_array[p1[0]][p1[1]] >= threshold) {
                    if (dis_array[p2[0]][p2[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
                        is_add = false;
                    }
                }
                if (dis_array[p2[0]][p2[1]] >= threshold) {
                    if (dis_array[p1[0]][p1[1]] >= 0.5 * dis_array[p[0]][p[1]]) {
                        is_add = false;
                    }
                }
                if (dis_array[p1[0]][p1[1]] >= threshold) {
                    if (dis_array[p2[0]][p2[1]] >= threshold) {
                        is_add = false;
                    }
                }
                if (dis_array[p2[0]][p2[1]] >= threshold) {
                    if (dis_array[p1[0]][p1[1]] >= threshold) {
                        is_add = false;
                    }
                }
            }
        }
        if (is_add) {
            const double px = mean_x_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
                              img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
            const double py = mean_y_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
                              img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
            const Eigen::Vector3d coord = py * x_axis + px * y_axis + project_center;
            pcl::PointXYZ pi;
            pi.x = coord[0];
            pi.y = coord[1];
            pi.z = coord[2];
            BinaryDescriptor single_binary =
                binary_container[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
            single_binary.location_ = coord;
            binary_list.emplace_back(single_binary);
        }
    }

    for (int i = 0; i < x_axis_len; i++) {
        delete[] binary_container[i];
        delete[] dis_container[i];
        delete[] img_count[i];
        delete[] dis_array[i];
        delete[] mean_x_list[i];
        delete[] mean_y_list[i];
    }
    delete[] binary_container;
    delete[] dis_container;
    delete[] img_count;
    delete[] dis_array;
    delete[] mean_x_list;
    delete[] mean_y_list;
}

void BtcDescManager::non_maxi_suppression(std::vector<BinaryDescriptor> &binary_list) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr prepare_key_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
    std::vector<int> pre_count_list;
    prepare_key_cloud->reserve(binary_list.size());
    pre_count_list.reserve(binary_list.size());
    std::vector<bool> is_add_list(binary_list.size(), true);
    for (auto var : binary_list) {
        pcl::PointXYZ pi;
        pi.x = var.location_[0];
        pi.y = var.location_[1];
        pi.z = var.location_[2];
        prepare_key_cloud->emplace_back(pi);
        pre_count_list.emplace_back(var.summary_);
    }
    kd_tree.setInputCloud(prepare_key_cloud);
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;
    const double radius = config_setting_.non_max_suppression_radius_;
    for (size_t i = 0; i < prepare_key_cloud->size(); i++) {
        const pcl::PointXYZ &searchPoint = prepare_key_cloud->points[i];
        if (kd_tree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch,
                                 pointRadiusSquaredDistance) > 0) {
            for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j) {
                if (pointIdxRadiusSearch[j] == i) {
                    continue;
                }
                if (pre_count_list[i] <= pre_count_list[pointIdxRadiusSearch[j]]) {
                    is_add_list[i] = false;
                }
            }
        }
    }
    std::vector<BinaryDescriptor> pass_binary_list;
    pass_binary_list.reserve(is_add_list.size());
    for (size_t i = 0; i < is_add_list.size(); i++) {
        if (is_add_list[i]) {
            pass_binary_list.emplace_back(binary_list[i]);
        }
    }
    pass_binary_list.shrink_to_fit();

    binary_list.clear();
    binary_list.reserve(pass_binary_list.size());
    for (auto var : pass_binary_list) {
        binary_list.emplace_back(var);
    }
    return;
}

void BtcDescManager::generate_btc(const std::vector<BinaryDescriptor> &binary_list,
                                  const int &frame_id,
                                  std::vector<BTC> &btc_list) {
    const double scale = 1.0 / config_setting_.std_side_resolution_;
    std::unordered_map<VOXEL_LOC, bool> feat_map;
    pcl::PointCloud<pcl::PointXYZ> key_cloud;
    key_cloud.reserve(binary_list.size());
    for (auto var : binary_list) {
        pcl::PointXYZ pi;
        pi.x = var.location_[0];
        pi.y = var.location_[1];
        pi.z = var.location_[2];
        key_cloud.emplace_back(pi);
    }
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
    kd_tree->setInputCloud(key_cloud.makeShared());
    int K = config_setting_.descriptor_near_num_;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    for (size_t i = 0; i < key_cloud.size(); i++) {
        const auto &searchPoint = key_cloud.points[i];
        if (kd_tree->nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) >
            0) {
            for (int m = 1; m < K - 1; m++) {
                for (int n = m + 1; n < K; n++) {
                    const pcl::PointXYZ &p1 = searchPoint;
                    const pcl::PointXYZ &p2 = key_cloud.points[pointIdxNKNSearch[m]];
                    const pcl::PointXYZ &p3 = key_cloud.points[pointIdxNKNSearch[n]];
                    double a =
                        sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
                    double b =
                        sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2) + pow(p1.z - p3.z, 2));
                    double c =
                        sqrt(pow(p3.x - p2.x, 2) + pow(p3.y - p2.y, 2) + pow(p3.z - p2.z, 2));
                    if (a > config_setting_.descriptor_max_len_ ||
                        b > config_setting_.descriptor_max_len_ ||
                        c > config_setting_.descriptor_max_len_ ||
                        a < config_setting_.descriptor_min_len_ ||
                        b < config_setting_.descriptor_min_len_ ||
                        c < config_setting_.descriptor_min_len_) {
                        continue;
                    }
                    double temp;
                    Eigen::Vector3d A, B, C;
                    Eigen::Vector3i l1, l2, l3;
                    Eigen::Vector3i l_temp;
                    l1 << 1, 2, 0;
                    l2 << 1, 0, 3;
                    l3 << 0, 2, 3;
                    if (a > b) {
                        temp = a;
                        a = b;
                        b = temp;
                        l_temp = l1;
                        l1 = l2;
                        l2 = l_temp;
                    }
                    if (b > c) {
                        temp = b;
                        b = c;
                        c = temp;
                        l_temp = l2;
                        l2 = l3;
                        l3 = l_temp;
                    }
                    if (a > b) {
                        temp = a;
                        a = b;
                        b = temp;
                        l_temp = l1;
                        l1 = l2;
                        l2 = l_temp;
                    }
                    if (fabs(c - (a + b)) < 0.2) {
                        continue;
                    }

                    pcl::PointXYZ d_p;
                    d_p.x = a * 1000;
                    d_p.y = b * 1000;
                    d_p.z = c * 1000;
                    VOXEL_LOC position((int64_t)d_p.x, (int64_t)d_p.y, (int64_t)d_p.z);
                    auto iter = feat_map.find(position);
                    Eigen::Vector3d normal_1, normal_2, normal_3;
                    BinaryDescriptor binary_A;
                    BinaryDescriptor binary_B;
                    BinaryDescriptor binary_C;
                    if (iter == feat_map.end()) {
                        if (l1[0] == l2[0]) {
                            A << p1.x, p1.y, p1.z;
                            binary_A = binary_list[i];
                        } else if (l1[1] == l2[1]) {
                            A << p2.x, p2.y, p2.z;
                            binary_A = binary_list[pointIdxNKNSearch[m]];
                        } else {
                            A << p3.x, p3.y, p3.z;
                            binary_A = binary_list[pointIdxNKNSearch[n]];
                        }
                        if (l1[0] == l3[0]) {
                            B << p1.x, p1.y, p1.z;
                            binary_B = binary_list[i];
                        } else if (l1[1] == l3[1]) {
                            B << p2.x, p2.y, p2.z;
                            binary_B = binary_list[pointIdxNKNSearch[m]];
                        } else {
                            B << p3.x, p3.y, p3.z;
                            binary_B = binary_list[pointIdxNKNSearch[n]];
                        }
                        if (l2[0] == l3[0]) {
                            C << p1.x, p1.y, p1.z;
                            binary_C = binary_list[i];
                        } else if (l2[1] == l3[1]) {
                            C << p2.x, p2.y, p2.z;
                            binary_C = binary_list[pointIdxNKNSearch[m]];
                        } else {
                            C << p3.x, p3.y, p3.z;
                            binary_C = binary_list[pointIdxNKNSearch[n]];
                        }
                        BTC single_descriptor;
                        single_descriptor.binary_A_ = binary_A;
                        single_descriptor.binary_B_ = binary_B;
                        single_descriptor.binary_C_ = binary_C;
                        single_descriptor.center_ = (A + B + C) / 3;
                        single_descriptor.triangle_ << scale * a, scale * b, scale * c;
                        single_descriptor.angle_[0] = fabs(5 * normal_1.dot(normal_2));
                        single_descriptor.angle_[1] = fabs(5 * normal_1.dot(normal_3));
                        single_descriptor.angle_[2] = fabs(5 * normal_3.dot(normal_2));
                        single_descriptor.frame_number_ = frame_id;
                        btc_list.emplace_back(single_descriptor);
                        feat_map[position] = true;
                    }
                }
            }
        }
    }
}

void BtcDescManager::candidate_selector(const std::vector<BTC> &current_STD_list,
                                        std::vector<BTCMatchList> &candidate_matcher_vec) {
    std::vector<Eigen::Vector3i> voxel_round;
    voxel_round.reserve(27);
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            for (int z = -1; z <= 1; z++) {
                voxel_round.emplace_back(x, y, z);
            }
        }
    }
    std::vector<bool> useful_match(current_STD_list.size(), false);
    std::vector<std::vector<size_t>> useful_match_index(current_STD_list.size());
    std::vector<std::vector<BTC_LOC>> useful_match_position(current_STD_list.size());

#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (size_t i = 0; i < current_STD_list.size(); i++) {
        const auto &descriptor = current_STD_list[i];
        BTC_LOC position;
        BTC_LOC best_position;
        const double dis_threshold =
            descriptor.triangle_.norm() * config_setting_.rough_dis_threshold_;  // old 0.005
        for (auto voxel_inc : voxel_round) {
            position.x = (int)(descriptor.triangle_[0] + voxel_inc[0]);
            position.y = (int)(descriptor.triangle_[1] + voxel_inc[1]);
            position.z = (int)(descriptor.triangle_[2] + voxel_inc[2]);
            const Eigen::Vector3d voxel_center((double)position.x + 0.5, (double)position.y + 0.5,
                                               (double)position.z + 0.5);
            if ((descriptor.triangle_ - voxel_center).norm() < 1.5) {
                auto iter = data_base_.find(position);
                if (iter != data_base_.end()) {
                    for (size_t j = 0; j < data_base_[position].size(); j++) {
                        if ((descriptor.frame_number_ - data_base_[position][j].frame_number_) >
                            config_setting_.skip_near_num_) {
                            const double dis =
                                (descriptor.triangle_ - data_base_[position][j].triangle_).norm();
                            if (dis < dis_threshold) {
                                const double similarity =
                                    (binary_similarity(descriptor.binary_A_,
                                                       data_base_[position][j].binary_A_) +
                                     binary_similarity(descriptor.binary_B_,
                                                       data_base_[position][j].binary_B_) +
                                     binary_similarity(descriptor.binary_C_,
                                                       data_base_[position][j].binary_C_)) /
                                    3;
                                if (similarity > config_setting_.similarity_threshold_) {
                                    useful_match[i] = true;
                                    useful_match_position[i].emplace_back(position);
                                    useful_match_index[i].emplace_back(j);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    double match_array[20000] = {0};
    std::vector<int> match_list_index;
    std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>> index_recorder;
    for (size_t i = 0; i < useful_match.size(); i++) {
        if (useful_match[i]) {
            for (size_t j = 0; j < useful_match_index[i].size(); j++) {
                match_array[data_base_[useful_match_position[i][j]][useful_match_index[i][j]]
                                .frame_number_] += 1;
                index_recorder.emplace_back(i, j);
                match_list_index.emplace_back(
                    data_base_[useful_match_position[i][j]][useful_match_index[i][j]]
                        .frame_number_);
            }
        }
    }

    // use index recorder
    for (int cnt = 0; cnt < config_setting_.candidate_num_; cnt++) {
        double max_vote = 1;
        int max_vote_index = -1;
        for (int i = 0; i < 20000; i++) {
            if (match_array[i] > max_vote) {
                max_vote = match_array[i];
                max_vote_index = i;
            }
        }
        if (max_vote_index >= 0 && max_vote >= 5) {
            BTCMatchList match_triangle_list;
            match_triangle_list.match_list_.reserve(index_recorder.size());
            match_array[max_vote_index] = 0;
            match_triangle_list.match_frame_ = max_vote_index;
            match_triangle_list.match_id_.first = current_frame_id_;
            match_triangle_list.match_id_.second = max_vote_index;
            for (size_t i = 0; i < index_recorder.size(); i++) {
                if (match_list_index[i] == max_vote_index) {
                    std::pair<BTC, BTC> single_match_pair;
                    single_match_pair.first = current_STD_list[index_recorder[i][0]];
                    single_match_pair.second =
                        data_base_[useful_match_position[index_recorder[i][0]]
                                                        [index_recorder[i][1]]]
                                  [useful_match_index[index_recorder[i][0]][index_recorder[i][1]]];
                    match_triangle_list.match_list_.emplace_back(single_match_pair);
                }
            }
            candidate_matcher_vec.emplace_back(match_triangle_list);
        }
    }
}

void BtcDescManager::candidate_verify(const BTCMatchList &candidate_matcher,
                                      double &verify_score,
                                      std::pair<Eigen::Vector3d, Eigen::Matrix3d> &relative_pose,
                                      std::vector<std::pair<BTC, BTC>> &sucess_match_list) {
    const double dis_threshold = 3.0;
    const int skip_len = (int)(candidate_matcher.match_list_.size() / 50) + 1;
    const int use_size = candidate_matcher.match_list_.size() / skip_len;
    std::vector<int> vote_list(use_size);
    std::mutex mylock;

#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (size_t i = 0; i < use_size; i++) {
        const auto &single_pair = candidate_matcher.match_list_[i * skip_len];
        int vote = 0;
        Eigen::Matrix3d test_rot;
        Eigen::Vector3d test_t;
        triangle_solver(single_pair, test_t, test_rot);
        std::for_each(
            candidate_matcher.match_list_.cbegin(), candidate_matcher.match_list_.cend(),
            [&](const auto &verify_pair) {
                const Eigen::Vector3d A = verify_pair.first.binary_A_.location_;
                const Eigen::Vector3d A_transform = test_rot * A + test_t;
                const Eigen::Vector3d B = verify_pair.first.binary_B_.location_;
                const Eigen::Vector3d B_transform = test_rot * B + test_t;
                const Eigen::Vector3d C = verify_pair.first.binary_C_.location_;
                const Eigen::Vector3d C_transform = test_rot * C + test_t;
                const double dis_A = (A_transform - verify_pair.second.binary_A_.location_).norm();
                const double dis_B = (B_transform - verify_pair.second.binary_B_.location_).norm();
                const double dis_C = (C_transform - verify_pair.second.binary_C_.location_).norm();
                if (dis_A < dis_threshold && dis_B < dis_threshold && dis_C < dis_threshold) {
                    vote++;
                }
            });
        mylock.lock();
        vote_list[i] = vote;
        mylock.unlock();
    }

    const auto max_vote_iter = std::max_element(vote_list.cbegin(), vote_list.cend());
    int max_vote_index = std::distance(vote_list.cbegin(), max_vote_iter);
    int max_vote = *max_vote_iter;
    if (max_vote >= 4) {
        const auto &best_pair = candidate_matcher.match_list_[max_vote_index * skip_len];
        Eigen::Matrix3d best_rot;
        Eigen::Vector3d best_t;
        triangle_solver(best_pair, best_t, best_rot);
        relative_pose.first = best_t;
        relative_pose.second = best_rot;
        sucess_match_list.clear();
        sucess_match_list.reserve(candidate_matcher.match_list_.size());
        std::for_each(
            candidate_matcher.match_list_.cbegin(), candidate_matcher.match_list_.cend(),
            [&](const auto &verify_pair) {
                const Eigen::Vector3d A = verify_pair.first.binary_A_.location_;
                const Eigen::Vector3d A_transform = best_rot * A + best_t;
                const Eigen::Vector3d B = verify_pair.first.binary_B_.location_;
                const Eigen::Vector3d B_transform = best_rot * B + best_t;
                const Eigen::Vector3d C = verify_pair.first.binary_C_.location_;
                const Eigen::Vector3d C_transform = best_rot * C + best_t;
                const double dis_A = (A_transform - verify_pair.second.binary_A_.location_).norm();
                const double dis_B = (B_transform - verify_pair.second.binary_B_.location_).norm();
                const double dis_C = (C_transform - verify_pair.second.binary_C_.location_).norm();
                if (dis_A < dis_threshold && dis_B < dis_threshold && dis_C < dis_threshold) {
                    sucess_match_list.emplace_back(verify_pair);
                }
            });
        sucess_match_list.shrink_to_fit();
        verify_score = plane_geometric_verify(plane_cloud_vec_.back(),
                                              plane_cloud_vec_[candidate_matcher.match_id_.second],
                                              relative_pose);
    } else {
        verify_score = -1;
    }
}

void BtcDescManager::triangle_solver(const std::pair<BTC, BTC> &std_pair,
                                     Eigen::Vector3d &t,
                                     Eigen::Matrix3d &rot) {
    Eigen::Matrix3d src = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d ref = Eigen::Matrix3d::Zero();
    src.col(0) = std_pair.first.binary_A_.location_ - std_pair.first.center_;
    src.col(1) = std_pair.first.binary_B_.location_ - std_pair.first.center_;
    src.col(2) = std_pair.first.binary_C_.location_ - std_pair.first.center_;
    ref.col(0) = std_pair.second.binary_A_.location_ - std_pair.second.center_;
    ref.col(1) = std_pair.second.binary_B_.location_ - std_pair.second.center_;
    ref.col(2) = std_pair.second.binary_C_.location_ - std_pair.second.center_;
    Eigen::Matrix3d covariance = src * ref.transpose();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::Matrix3d V = svd.matrixV();
    const Eigen::Matrix3d U = svd.matrixU();
    rot = V * U.transpose();
    if (rot.determinant() < 0) {
        Eigen::Matrix3d K;
        K << 1, 0, 0, 0, 1, 0, 0, 0, -1;
        rot = V * K * U.transpose();
    }
    t = -rot * std_pair.first.center_ + std_pair.second.center_;
}

double BtcDescManager::plane_geometric_verify(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
    const std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform) {
    const Eigen::Vector3d t = transform.first;
    const Eigen::Matrix3d rot = transform.second;
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    input_cloud->reserve(target_cloud->size());
    for (size_t i = 0; i < target_cloud->size(); i++) {
        pcl::PointXYZ pi;
        pi.x = target_cloud->points[i].x;
        pi.y = target_cloud->points[i].y;
        pi.z = target_cloud->points[i].z;
        input_cloud->emplace_back(pi);
    }

    kd_tree->setInputCloud(input_cloud);
    std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);
    double useful_match = 0;
    const double normal_threshold = config_setting_.normal_threshold_;
    const double dis_threshold = config_setting_.dis_threshold_;
    std::for_each(source_cloud->cbegin(), source_cloud->cend(), [&](const auto &searchPoint) {
        Eigen::Vector3d pi = point2vec(searchPoint);
        Eigen::Vector3d ni = normal2vec(searchPoint);
        pi = rot * pi + t;
        ni = rot * ni;
        pcl::PointXYZ use_search_point;
        use_search_point.x = pi[0];
        use_search_point.y = pi[1];
        use_search_point.z = pi[2];
        if (kd_tree->nearestKSearch(use_search_point, 1, pointIdxNKNSearch,
                                    pointNKNSquaredDistance) > 0) {
            pcl::PointXYZINormal nearestPoint = target_cloud->points[pointIdxNKNSearch[0]];
            const Eigen::Vector3d tpi = point2vec(nearestPoint);
            const Eigen::Vector3d tni = normal2vec(nearestPoint);
            const Eigen::Vector3d normal_inc = ni - tni;
            const Eigen::Vector3d normal_add = ni + tni;
            const double point_to_plane = fabs(tni.transpose() * (pi - tpi));
            if ((normal_inc.norm() < normal_threshold || normal_add.norm() < normal_threshold) &&
                point_to_plane < dis_threshold) {
                useful_match++;
            }
        }
    });
    return useful_match / source_cloud->size();
}

int BtcDescManager::ProcessNewScan(const std::vector<Eigen::Vector3d> &pcl) {
    const pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud = EigenToPCL(pcl);

    std::vector<BTC> btc_vec;
    this->GenerateBTCDescs(current_cloud, btc_vec);

    if (keyCloudInd > config_setting_.skip_near_num_) {
        this->SearchLoop(btc_vec);
    }
    this->AddBTCDescs(btc_vec);
    keyCloudInd++;
    return loop_match_ids_.size();
}

void BtcDescManager::AddToDatabase(const std::vector<Eigen::Vector3d> &pcl) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud = EigenToPCL(pcl);

    std::vector<BTC> btc_vec;
    this->GenerateBTCDescs(current_cloud, btc_vec);
    this->AddBTCDescs(btc_vec);
    keyCloudInd++;
}

int BtcDescManager::ComputeClosure(const std::vector<Eigen::Vector3d> &pcl) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud = EigenToPCL(pcl);

    std::vector<BTC> btc_vec;
    this->GenerateBTCDescs(current_cloud, btc_vec);
    this->SearchLoop(btc_vec);
    return loop_match_ids_.size();
}

std::tuple<int, double, Eigen::Vector3d, Eigen::Matrix3d> BtcDescManager::GetClosureDataAtIdx(
    const int idx) {
    return {loop_match_ids_[idx], loop_match_scores_[idx], loop_trs_[idx], loop_rots_[idx]};
}
