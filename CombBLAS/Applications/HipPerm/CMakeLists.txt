ADD_EXECUTABLE(stream_experiment stream_experiment_all.cpp)
ADD_EXECUTABLE(permutation2 permutation2.cpp)
ADD_EXECUTABLE(extract extract.cpp)
ADD_EXECUTABLE(assign assign.cpp)
ADD_EXECUTABLE(assignWithPermute assignWithPermute.cpp)


TARGET_LINK_LIBRARIES( stream_experiment CombBLAS )
TARGET_LINK_LIBRARIES( permutation2 CombBLAS )
TARGET_LINK_LIBRARIES( extract CombBLAS )
TARGET_LINK_LIBRARIES( assign CombBLAS )
TARGET_LINK_LIBRARIES( assignWithPermute CombBLAS )