#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct Array
{
    std::vector<int> shape;
    std::vector<float> data;                         // row-major, last index fastest
};

inline Array load_npy(const std::string& path)
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

    if (header.find("'descr': '<f4'") == std::string::npos) fail("expected float32");
    if (header.find("'fortran_order': False") == std::string::npos) fail("expected row-major");

    size_t s = header.find("'shape': (");
    if (s == std::string::npos) fail("no shape");
    Array a;
    size_t count = 1;
    for (const char* c = header.c_str() + s + 10; *c != ')';)  // e.g. "(3, 3, 1, 64)" or "(64,)"
    {
        char* end;
        long n = strtol(c, &end, 10);
        if (end == c) break;
        a.shape.push_back(int(n));
        count *= size_t(n);
        c = end;
        while (*c == ',' || *c == ' ') ++c;
    }
    if (a.shape.empty()) fail("empty shape");

    a.data.resize(count);
    f.read(reinterpret_cast<char*>(a.data.data()), count * sizeof(float));
    if (!f) fail("file too short");
    return a;
}
