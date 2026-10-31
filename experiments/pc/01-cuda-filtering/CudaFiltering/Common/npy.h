#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct ImageD
{
    int W = 0, H = 0;
    std::vector<double> data;                        // row-major: data[v * W + u]
};

inline ImageD load_npy(const std::string& path)
{
    auto fail = [&](const char* why) { printf("%s: %s\n", path.c_str(), why); exit(1); };

    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open");

    char magic[8];
    f.read(magic, 8);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) fail("not an npy file");
    if (magic[6] != 1) fail("only npy version 1 supported");

    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);  // x86 is little-endian, same as the file
    std::string header(header_len, ' ');
    f.read(&header[0], header_len);

    if (header.find("'descr': '<f8'") == std::string::npos) fail("expected float64");
    if (header.find("'fortran_order': False") == std::string::npos) fail("expected row-major");

    size_t s = header.find("'shape': (");
    if (s == std::string::npos) fail("no shape");
    ImageD img;
    if (sscanf(header.c_str() + s, "'shape': (%d, %d)", &img.H, &img.W) != 2) fail("expected a 2D shape");

    img.data.resize(size_t(img.W) * img.H);
    f.read(reinterpret_cast<char*>(img.data.data()), img.data.size() * sizeof(double));
    if (!f) fail("file too short");
    return img;
}