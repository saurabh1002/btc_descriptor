add_library(btcdesc STATIC BTCdesc.cpp)
target_include_directories(btcdesc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(btcdesc PUBLIC ${PCL_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
set_global_target_properties(btcdesc)
