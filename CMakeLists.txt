cmake_minimum_required(VERSION 3.8)
project(feature_matching)

find_package(OpenCV REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
)