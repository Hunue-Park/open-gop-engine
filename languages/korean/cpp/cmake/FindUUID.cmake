find_path(UUID_INCLUDE_DIRS uuid.h
  HINTS
    /opt/homebrew/include
    /opt/homebrew/include/ossp
    /usr/include
    /usr/local/include
)

find_library(UUID_LIBRARIES
  NAMES ossp-uuid uuid
  PATHS
    /opt/homebrew/lib
    /usr/lib
    /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UUID DEFAULT_MSG UUID_LIBRARIES UUID_INCLUDE_DIRS)

mark_as_advanced(UUID_INCLUDE_DIRS UUID_LIBRARIES)
