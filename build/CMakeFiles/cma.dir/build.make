# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nima/ML/cma2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nima/ML/cma2/build

# Include any dependencies generated for this target.
include CMakeFiles/cma.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cma.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cma.dir/flags.make

CMakeFiles/cma.dir/wrapper.cpp.o: CMakeFiles/cma.dir/flags.make
CMakeFiles/cma.dir/wrapper.cpp.o: ../wrapper.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nima/ML/cma2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cma.dir/wrapper.cpp.o"
	/home/nima/anaconda3/envs/rga/bin/x86_64-conda_cos6-linux-gnu-c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cma.dir/wrapper.cpp.o -c /home/nima/ML/cma2/wrapper.cpp

CMakeFiles/cma.dir/wrapper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cma.dir/wrapper.cpp.i"
	/home/nima/anaconda3/envs/rga/bin/x86_64-conda_cos6-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nima/ML/cma2/wrapper.cpp > CMakeFiles/cma.dir/wrapper.cpp.i

CMakeFiles/cma.dir/wrapper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cma.dir/wrapper.cpp.s"
	/home/nima/anaconda3/envs/rga/bin/x86_64-conda_cos6-linux-gnu-c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nima/ML/cma2/wrapper.cpp -o CMakeFiles/cma.dir/wrapper.cpp.s

# Object files for target cma
cma_OBJECTS = \
"CMakeFiles/cma.dir/wrapper.cpp.o"

# External object files for target cma
cma_EXTERNAL_OBJECTS =

cma.cpython-38-x86_64-linux-gnu.so: CMakeFiles/cma.dir/wrapper.cpp.o
cma.cpython-38-x86_64-linux-gnu.so: CMakeFiles/cma.dir/build.make
cma.cpython-38-x86_64-linux-gnu.so: CMakeFiles/cma.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nima/ML/cma2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module cma.cpython-38-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cma.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cma.dir/build: cma.cpython-38-x86_64-linux-gnu.so

.PHONY : CMakeFiles/cma.dir/build

CMakeFiles/cma.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cma.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cma.dir/clean

CMakeFiles/cma.dir/depend:
	cd /home/nima/ML/cma2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nima/ML/cma2 /home/nima/ML/cma2 /home/nima/ML/cma2/build /home/nima/ML/cma2/build /home/nima/ML/cma2/build/CMakeFiles/cma.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cma.dir/depend

