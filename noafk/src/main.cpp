/*
 * NoAFK - World of Warcraft Anti-AFK Application
 * Prevents character timeout by sending randomized dummy inputs
 * Windows System Tray Application
 */

#define WIN32_LEAN_AND_MEAN
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif

#include <windows.h>
#include <shellapi.h>
#include <commctrl.h>
#include <random>
#include <string>
#include <chrono>

#include "resource.h"
#include "settings.h"
#include "antiafk.h"

// MSVC only
#ifdef _MSC_VER
#pragma comment(lib, "comctl32.lib")
#pragma comment(linker, "/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
#endif

// Window messages
#define WM_TRAYICON (WM_USER + 1)
#define WM_UPDATESTATUS (WM_USER + 2)

// Menu IDs
#define ID_TRAY_SETTINGS    1001
#define ID_TRAY_PAUSE       1002
#define ID_TRAY_RESUME      1003
#define ID_TRAY_EXIT        1004
#define ID_TRAY_STATUS      1005

// Global variables
HWND g_hWnd = nullptr;
HINSTANCE g_hInstance = nullptr;
NOTIFYICONDATA g_nid = {};
Settings g_settings;
AntiAFK g_antiAFK;
bool g_isPaused = false;

// Forward declarations
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
void ShowTrayMenu(HWND hWnd);
void InitTrayIcon(HWND hWnd);
void UpdateTrayIcon();
void ShowStartupNotification();
void CleanupTrayIcon();

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    (void)hPrevInstance;
    (void)lpCmdLine;
    (void)nCmdShow;

    g_hInstance = hInstance;

    // Initialize common controls
    INITCOMMONCONTROLSEX icex = {};
    icex.dwSize = sizeof(icex);
    icex.dwICC = ICC_STANDARD_CLASSES;
    InitCommonControlsEx(&icex);

    // Register window class
    WNDCLASSEX wc = {};
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = L"NoAFKClass";
    wc.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APPICON));
    wc.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_APPICON));

    if (!RegisterClassEx(&wc)) {
        MessageBox(nullptr, L"Failed to register window class!", L"Error", MB_ICONERROR);
        return 1;
    }

    // Create hidden window for message processing
    g_hWnd = CreateWindowEx(
        0,
        L"NoAFKClass",
        L"NoAFK",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        400, 300,
        nullptr,
        nullptr,
        hInstance,
        nullptr
    );

    if (!g_hWnd) {
        MessageBox(nullptr, L"Failed to create window!", L"Error", MB_ICONERROR);
        return 1;
    }

    // Load settings
    g_settings.Load();

    // Initialize tray icon
    InitTrayIcon(g_hWnd);

    // Show startup notification
    ShowStartupNotification();

    // Initialize and start anti-AFK
    g_antiAFK.SetSettings(&g_settings);
    g_antiAFK.Start(g_hWnd);

    // Message loop
    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    // Cleanup
    g_antiAFK.Stop();
    CleanupTrayIcon();

    return (int)msg.wParam;
}

void InitTrayIcon(HWND hWnd) {
    g_nid.cbSize = sizeof(NOTIFYICONDATA);
    g_nid.hWnd = hWnd;
    g_nid.uID = 1;
    g_nid.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP | NIF_INFO;
    g_nid.uCallbackMessage = WM_TRAYICON;
    g_nid.hIcon = LoadIcon(g_hInstance, MAKEINTRESOURCE(IDI_APPICON));
    wcscpy_s(g_nid.szTip, L"NoAFK - Active");
    g_nid.dwInfoFlags = NIIF_INFO;

    Shell_NotifyIcon(NIM_ADD, &g_nid);
}

void ShowStartupNotification() {
    g_nid.uFlags = NIF_INFO;
    wcscpy_s(g_nid.szInfoTitle, L"NoAFK Started");
    wcscpy_s(g_nid.szInfo, L"Application is now running in the system tray.\nRight-click the icon for options.");
    g_nid.dwInfoFlags = NIIF_INFO;
    Shell_NotifyIcon(NIM_MODIFY, &g_nid);
}

void UpdateTrayIcon() {
    if (g_isPaused) {
        wcscpy_s(g_nid.szTip, L"NoAFK - Paused");
        g_nid.hIcon = LoadIcon(g_hInstance, MAKEINTRESOURCE(IDI_PAUSED));
    } else {
        wcscpy_s(g_nid.szTip, L"NoAFK - Active");
        g_nid.hIcon = LoadIcon(g_hInstance, MAKEINTRESOURCE(IDI_APPICON));
    }
    g_nid.uFlags = NIF_ICON | NIF_TIP;
    Shell_NotifyIcon(NIM_MODIFY, &g_nid);
}

void CleanupTrayIcon() {
    Shell_NotifyIcon(NIM_DELETE, &g_nid);
}

void ShowTrayMenu(HWND hWnd) {
    POINT pt;
    GetCursorPos(&pt);

    HMENU hMenu = CreatePopupMenu();

    // Status item (disabled, just for display)
    std::wstring status = g_isPaused ? L"Status: Paused" : L"Status: Active";
    if (!g_isPaused && g_antiAFK.IsWoWRunning()) {
        status += L" (WoW Found)";
    } else if (!g_isPaused) {
        status += L" (WoW Not Found)";
    }
    AppendMenu(hMenu, MF_STRING | MF_DISABLED, ID_TRAY_STATUS, status.c_str());
    AppendMenu(hMenu, MF_SEPARATOR, 0, nullptr);

    AppendMenu(hMenu, MF_STRING, ID_TRAY_SETTINGS, L"Settings...");
    AppendMenu(hMenu, MF_SEPARATOR, 0, nullptr);

    if (g_isPaused) {
        AppendMenu(hMenu, MF_STRING, ID_TRAY_RESUME, L"Resume");
    } else {
        AppendMenu(hMenu, MF_STRING, ID_TRAY_PAUSE, L"Pause");
    }

    AppendMenu(hMenu, MF_SEPARATOR, 0, nullptr);
    AppendMenu(hMenu, MF_STRING, ID_TRAY_EXIT, L"Exit");

    // Required for proper menu behavior
    SetForegroundWindow(hWnd);

    TrackPopupMenu(hMenu, TPM_RIGHTALIGN | TPM_BOTTOMALIGN | TPM_RIGHTBUTTON,
                   pt.x, pt.y, 0, hWnd, nullptr);

    PostMessage(hWnd, WM_NULL, 0, 0);
    DestroyMenu(hMenu);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
        case WM_TRAYICON:
            if (lParam == WM_RBUTTONUP || lParam == WM_LBUTTONUP) {
                ShowTrayMenu(hWnd);
            }
            break;

        case WM_COMMAND:
            switch (LOWORD(wParam)) {
                case ID_TRAY_SETTINGS:
                    ShowSettingsDialog(g_hInstance, hWnd, &g_settings);
                    g_antiAFK.SetSettings(&g_settings);
                    break;

                case ID_TRAY_PAUSE:
                    g_isPaused = true;
                    g_antiAFK.Pause();
                    UpdateTrayIcon();
                    break;

                case ID_TRAY_RESUME:
                    g_isPaused = false;
                    g_antiAFK.Resume();
                    UpdateTrayIcon();
                    break;

                case ID_TRAY_EXIT:
                    PostQuitMessage(0);
                    break;
            }
            break;

        case WM_DESTROY:
            PostQuitMessage(0);
            break;

        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}
