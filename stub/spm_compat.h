#pragma once
// Homebrew's sentencepiece returns a string_view from Status::message(); the
// build QVAC uses returns std::string. Bridge that one concatenation for this
// local benchmark only (adding to namespace std is not portable code).
#include <string>
#include <string_view>
#include <cassert>  // nmt_graph_decoder.cpp uses assert without including it
namespace std {
inline string operator+(string a, string_view b) { a.append(b.data(), b.size()); return a; }
}
