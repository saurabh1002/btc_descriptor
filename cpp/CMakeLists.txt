add_library(btcdesc STATIC btc.cpp)
target_include_directories(btcdesc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(btcdesc PUBLIC ${PCL_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_compile_options(btcdesc PUBLIC -fPIC)
