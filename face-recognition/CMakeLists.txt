
file(GLOB faceRecognitionSources *.cpp)
file(GLOB faceRecognitionIncludes *.h )

cuda_add_executable(face-recognition ${faceRecognitionSources})
target_link_libraries(face-recognition nvcaffe_parser nvinfer jetson-inference)
