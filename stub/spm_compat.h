#pragma once
// Homebrew's sentencepiece returns a string_view from Status::message(); the
// build QVAC uses returns std::string. Bridge that one concatenation for this
// local benchmark only (adding to namespace std is not portable code).
#include <string>
#include <string_view>
// QVAC's NMT sources rely on these arriving through other headers (true with
// Homebrew's sentencepiece/abseil, not with GCC or MinGW), so include them here.
#include <cassert>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>
#include <memory>
namespace std {
inline string operator+(string a, string_view b) { a.append(b.data(), b.size()); return a; }
}
