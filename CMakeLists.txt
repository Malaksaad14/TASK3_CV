cmake_minimum_required(VERSION 3.10)
project(FeatureExtractor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Force use of the MSYS2 compiler (where wxWidgets is installed)
set(CMAKE_C_COMPILER "C:/msys64/ucrt64/bin/gcc.exe")
set(CMAKE_CXX_COMPILER "C:/msys64/ucrt64/bin/g++.exe")



include_directories(
    "C:/msys64/ucrt64/lib/wx/include/msw-unicode-3.2"
    "C:/msys64/ucrt64/include/wx-3.2"
)

add_definitions(-D_FILE_OFFSET_BITS=64 -DWXUSINGDLL -D__WXMSW__)

file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "src/*.h")

add_executable(${PROJECT_NAME} WIN32 ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} "-L C:/msys64/ucrt64/lib" -lwx_mswu_core-3.2 -lwx_baseu-3.2 -mwindows)
