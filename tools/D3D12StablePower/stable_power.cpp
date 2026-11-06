// Holds an NVIDIA GPU in D3D12 stable power state (fixed base clock) until Enter is pressed.
// The clock lock is GPU-wide, so CUDA programs started meanwhile run at the same fixed clock.
// Requires Windows Developer Mode.
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cstdio>
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

const UINT s_nvidia = 0x10DE;                        // PCI vendor id

int main()
{
    IDXGIFactory6* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) { printf("no DXGI factory\n"); return 1; }

    // first NVIDIA adapter; on laptops the default adapter can be the integrated GPU
    IDXGIAdapter1* adapter = nullptr;
    DXGI_ADAPTER_DESC1 desc = {};
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i)
    {
        adapter->GetDesc1(&desc);
        if (desc.VendorId == s_nvidia) break;
        adapter->Release();
        adapter = nullptr;
    }
    if (!adapter) { printf("no NVIDIA adapter\n"); return 1; }

    ID3D12Device* device = nullptr;
    if (FAILED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) { printf("no D3D12 device\n"); return 1; }

    HRESULT hr = device->SetStablePowerState(TRUE);
    if (FAILED(hr)) { printf("SetStablePowerState failed: 0x%08lx (Developer Mode off?)\n", hr); return 1; }
    printf("%ls: stable power state on. Press Enter to release.\n", desc.Description);
    getchar();

    device->Release();                               // releasing the device restores normal clocks
    adapter->Release();
    factory->Release();
    return 0;
}
