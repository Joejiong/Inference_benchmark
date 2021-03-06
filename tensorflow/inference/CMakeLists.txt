cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
add_executable(image_classification image_classification.cpp)
add_executable(yolov3 yolov3.cpp)

find_package(TensorflowCC REQUIRED)
target_link_libraries(image_classification TensorflowCC::TensorflowCC)
target_link_libraries(yolov3 TensorflowCC::TensorflowCC)

# link cuda if it is available
find_package(CUDA)
if(CUDA_FOUND)
  target_link_libraries(image_classification ${CUDA_LIBRARIES})
  target_link_libraries(yolov3 ${CUDA_LIBRARIES})
endif()
