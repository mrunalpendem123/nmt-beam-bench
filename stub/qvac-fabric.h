#pragma once
// Stand-in for @qvac/fabric's header: ggml is linked statically here, so the
// CPU and Metal backends are already registered and there is nothing to load.
inline const char* qvac_fabric_backends_dir() { return "(static)"; }
inline void qvac_fabric_load_backends() {}
