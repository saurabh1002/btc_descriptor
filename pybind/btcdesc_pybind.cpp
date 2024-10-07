// MIT License
//
// Copyright (c) 2023 Saurabh Gupta, Tiziano Guadagnino, Cyrill Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <Eigen/Core>
#include <vector>

#include "btc.h"
#include "stl_vector_eigen.h"

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);

namespace py = pybind11;
using namespace py::literals;

ConfigSetting GetConfigFromYAML(const py::dict &yaml_cfg)
{
    ConfigSetting config;

    // for binary descriptor
    config.useful_corner_num_ = yaml_cfg["useful_corner_num"].cast<int>();
    config.plane_merge_normal_thre_ = yaml_cfg["plane_merge_normal_thre"].cast<float>();
    config.plane_merge_dis_thre_ = yaml_cfg["plane_merge_dis_thre"].cast<float>();
    config.plane_detection_thre_ = yaml_cfg["plane_detection_thre"].cast<float>();
    config.voxel_size_ = yaml_cfg["voxel_size"].cast<float>();
    config.voxel_init_num_ = yaml_cfg["voxel_init_num"].cast<int>();
    config.proj_plane_num_ = yaml_cfg["proj_plane_num"].cast<int>();
    config.proj_image_resolution_ = yaml_cfg["proj_image_resolution"].cast<float>();
    config.proj_image_high_inc_ = yaml_cfg["proj_image_high_inc"].cast<float>();
    config.proj_dis_min_ = yaml_cfg["proj_dis_min"].cast<float>();
    config.proj_dis_max_ = yaml_cfg["proj_dis_max"].cast<float>();
    config.summary_min_thre_ = yaml_cfg["summary_min_thre"].cast<float>();
    config.line_filter_enable_ = yaml_cfg["line_filter_enable"].cast<int>();

    /* for STD */
    config.descriptor_near_num_ = yaml_cfg["descriptor_near_num"].cast<float>();
    config.descriptor_min_len_ = yaml_cfg["descriptor_min_len"].cast<float>();
    config.descriptor_max_len_ = yaml_cfg["descriptor_max_len"].cast<float>();
    config.non_max_suppression_radius_ = yaml_cfg["non_max_suppression_radius"].cast<float>();
    config.std_side_resolution_ = yaml_cfg["std_side_resolution"].cast<float>();

    /* for place recognition*/
    config.skip_near_num_ = yaml_cfg["skip_near_num"].cast<int>();
    config.candidate_num_ = yaml_cfg["candidate_num"].cast<int>();
    config.sub_frame_num_ = yaml_cfg["sub_frame_num"].cast<int>();
    config.rough_dis_threshold_ = yaml_cfg["rough_dis_threshold"].cast<float>();
    config.similarity_threshold_ = yaml_cfg["similarity_threshold"].cast<float>();
    config.icp_threshold_ = yaml_cfg["icp_threshold"].cast<float>();
    config.normal_threshold_ = yaml_cfg["normal_threshold"].cast<float>();
    config.dis_threshold_ = yaml_cfg["dis_threshold"].cast<float>();

    return config;
}

PYBIND11_MODULE(btcdesc_pybind, m)
{
    auto vector3dvector = pybind_eigen_vector_of_vector<Eigen::Vector3d>(
        m, "_VectorEigen3d", "std::vector<Eigen::Vector3d>",
        py::py_array_to_vectors_double<Eigen::Vector3d>);

    m.def("_VoxelDownSample", &down_sampling_voxel, "pl_feat"_a, "voxel_size"_a);

    py::class_<BtcDescManager> btcdesc(m, "_BtcDescManager", "");
    btcdesc
        .def(py::init([](const py::dict &cfg)
                      {
                 auto config = GetConfigFromYAML(cfg);
                 return BtcDescManager(config); }),
             "config"_a)
        .def("_ProcessNewScan", &BtcDescManager::ProcessNewScan, "pcl"_a)
        .def(
            "_GetClosureDataAtIdx",
            [](BtcDescManager &self, int idx)
            { return self.GetClosureDataAtIdx(idx); }, "idx"_a);
}
