# CMake generated Testfile for 
# Source directory: /home/ehassani/projects/private-combblas/CombBLAS/Applications
# Build directory: /home/ehassani/projects/private-combblas/CombBLAS/_build/Applications
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[BetwCent_Test]=] "/sw/eb/sw/impi/2021.13.0-intel-compilers-2024.2.0/mpi/2021.13/bin/mpiexec" "-n" "4" "/home/ehassani/projects/private-combblas/CombBLAS/_build/Applications/betwcent" "../TESTDATA/SCALE16BTW-TRANSBOOL/" "10" "96")
set_tests_properties([=[BetwCent_Test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/ehassani/projects/private-combblas/CombBLAS/Applications/CMakeLists.txt;20;ADD_TEST;/home/ehassani/projects/private-combblas/CombBLAS/Applications/CMakeLists.txt;0;")
add_test([=[TopDownBFS_Test]=] "/sw/eb/sw/impi/2021.13.0-intel-compilers-2024.2.0/mpi/2021.13/bin/mpiexec" "-n" "4" "/home/ehassani/projects/private-combblas/CombBLAS/_build/Applications/tdbfs" "Force" "17" "FastGen")
set_tests_properties([=[TopDownBFS_Test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/ehassani/projects/private-combblas/CombBLAS/Applications/CMakeLists.txt;21;ADD_TEST;/home/ehassani/projects/private-combblas/CombBLAS/Applications/CMakeLists.txt;0;")
add_test([=[DirOptBFS_Test]=] "/sw/eb/sw/impi/2021.13.0-intel-compilers-2024.2.0/mpi/2021.13/bin/mpiexec" "-n" "4" "/home/ehassani/projects/private-combblas/CombBLAS/_build/Applications/dobfs" "17")
set_tests_properties([=[DirOptBFS_Test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/ehassani/projects/private-combblas/CombBLAS/Applications/CMakeLists.txt;22;ADD_TEST;/home/ehassani/projects/private-combblas/CombBLAS/Applications/CMakeLists.txt;0;")
add_test([=[FBFS_Test]=] "/sw/eb/sw/impi/2021.13.0-intel-compilers-2024.2.0/mpi/2021.13/bin/mpiexec" "-n" "4" "/home/ehassani/projects/private-combblas/CombBLAS/_build/Applications/fbfs" "Gen" "16")
set_tests_properties([=[FBFS_Test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/ehassani/projects/private-combblas/CombBLAS/Applications/CMakeLists.txt;23;ADD_TEST;/home/ehassani/projects/private-combblas/CombBLAS/Applications/CMakeLists.txt;0;")
add_test([=[FMIS_Test]=] "/sw/eb/sw/impi/2021.13.0-intel-compilers-2024.2.0/mpi/2021.13/bin/mpiexec" "-n" "4" "/home/ehassani/projects/private-combblas/CombBLAS/_build/Applications/fmis" "17")
set_tests_properties([=[FMIS_Test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/ehassani/projects/private-combblas/CombBLAS/Applications/CMakeLists.txt;24;ADD_TEST;/home/ehassani/projects/private-combblas/CombBLAS/Applications/CMakeLists.txt;0;")
