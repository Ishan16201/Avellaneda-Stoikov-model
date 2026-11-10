#pragma once

#ifndef PROJECT3_GIT_REVISION
#define PROJECT3_GIT_REVISION "unknown"
#endif

#ifndef PROJECT3_VERSION
#define PROJECT3_VERSION "0.1.0"
#endif

namespace project3 {

[[nodiscard]] inline const char* git_revision() noexcept {
  return PROJECT3_GIT_REVISION;
}

[[nodiscard]] inline const char* library_version() noexcept {
  return PROJECT3_VERSION;
}

}  // namespace project3
