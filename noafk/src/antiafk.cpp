/*
 * NoAFK - Anti-AFK Engine Implementation
 * Core logic for preventing AFK timeout in World of Warcraft
 */

#include "antiafk.h"
#include <chrono>
#include <algorithm>

AntiAFK::AntiAFK() {
    // Seed random number generator
    std::random_device rd;
    m_rng.seed(rd());
}

AntiAFK::~AntiAFK() {
    Stop();
}

void AntiAFK::SetSettings(Settings* settings) {
    m_settings = settings;
}

void AntiAFK::Start(HWND notifyWnd) {
    if (m_running) return;

    m_notifyWnd = notifyWnd;
    m_running = true;
    m_paused = false;

    m_thread = std::thread(&AntiAFK::WorkerThread, this);
}

void AntiAFK::Stop() {
    m_running = false;
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

void AntiAFK::Pause() {
    m_paused = true;
}

void AntiAFK::Resume() {
    m_paused = false;
}

bool AntiAFK::IsWoWRunning() {
    return FindWoWWindow() != nullptr;
}

bool AntiAFK::IsWoWFocused() {
    HWND wowWnd = FindWoWWindow();
    if (!wowWnd) return false;
    return GetForegroundWindow() == wowWnd;
}

DWORD AntiAFK::GetTimeUntilNext() const {
    DWORD now = GetTickCount();
    if (now >= m_nextActionTime) return 0;
    return m_nextActionTime - now;
}

HWND AntiAFK::FindWoWWindow() {
    if (!m_settings) return nullptr;

    // First try to find by window class (more reliable)
    HWND wowWnd = FindWindow(L"GxWindowClass", nullptr);
    if (wowWnd) {
        // Verify it's the right process
        DWORD processId;
        GetWindowThreadProcessId(wowWnd, &processId);

        HANDLE hSnapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (hSnapshot != INVALID_HANDLE_VALUE) {
            PROCESSENTRY32W pe;
            pe.dwSize = sizeof(pe);

            if (Process32FirstW(hSnapshot, &pe)) {
                do {
                    if (pe.th32ProcessID == processId) {
                        std::wstring exeName = pe.szExeFile;
                        // Case-insensitive compare
                        std::transform(exeName.begin(), exeName.end(), exeName.begin(), ::towlower);
                        std::wstring targetName = m_settings->processName;
                        std::transform(targetName.begin(), targetName.end(), targetName.begin(), ::towlower);

                        if (exeName.find(targetName) != std::wstring::npos ||
                            targetName.find(exeName) != std::wstring::npos ||
                            exeName == L"wow.exe" || exeName == L"wowclassic.exe" ||
                            exeName == L"wow-64.exe" || exeName == L"wowt.exe" ||
                            exeName == L"wowb.exe") {
                            CloseHandle(hSnapshot);
                            return wowWnd;
                        }
                        break;
                    }
                } while (Process32NextW(hSnapshot, &pe));
            }
            CloseHandle(hSnapshot);
        }
    }

    // Fallback: enumerate all windows looking for WoW
    struct EnumData {
        HWND result;
        const wchar_t* processName;
    } enumData = { nullptr, m_settings->processName.c_str() };

    EnumWindows([](HWND hwnd, LPARAM lParam) -> BOOL {
        EnumData* data = (EnumData*)lParam;

        DWORD processId;
        GetWindowThreadProcessId(hwnd, &processId);

        HANDLE hProcess = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, processId);
        if (hProcess) {
            wchar_t exePath[MAX_PATH];
            DWORD size = MAX_PATH;
            if (QueryFullProcessImageNameW(hProcess, 0, exePath, &size)) {
                std::wstring path = exePath;
                std::transform(path.begin(), path.end(), path.begin(), ::towlower);

                if (path.find(L"wow") != std::wstring::npos) {
                    // Verify it's a visible game window
                    if (IsWindowVisible(hwnd)) {
                        wchar_t className[256];
                        GetClassName(hwnd, className, 256);
                        if (wcscmp(className, L"GxWindowClass") == 0) {
                            data->result = hwnd;
                            CloseHandle(hProcess);
                            return FALSE; // Stop enumeration
                        }
                    }
                }
            }
            CloseHandle(hProcess);
        }
        return TRUE; // Continue enumeration
    }, (LPARAM)&enumData);

    return enumData.result;
}

DWORD AntiAFK::GetRandomInterval() {
    if (!m_settings) return 60000; // Default 1 minute

    std::uniform_int_distribution<DWORD> dist(m_settings->minInterval, m_settings->maxInterval);
    return dist(m_rng);
}

void AntiAFK::WorkerThread() {
    // Initial delay before first action
    m_nextActionTime = GetTickCount() + GetRandomInterval();

    while (m_running) {
        // Sleep in small increments to allow responsive shutdown
        Sleep(100);

        if (!m_running) break;
        if (m_paused) continue;

        DWORD now = GetTickCount();
        if (now < m_nextActionTime) continue;

        // Time to send an input
        HWND wowWnd = FindWoWWindow();
        if (wowWnd) {
            // Check focus requirement
            if (m_settings && m_settings->onlyWhenFocused) {
                if (GetForegroundWindow() != wowWnd) {
                    // WoW not focused, skip this action but schedule next
                    m_nextActionTime = now + GetRandomInterval();
                    continue;
                }
            }

            SendAntiAFKInput(wowWnd);
        }

        // Schedule next action
        m_nextActionTime = GetTickCount() + GetRandomInterval();
    }
}

void AntiAFK::SendAntiAFKInput(HWND wowWnd) {
    if (!m_settings) return;

    int inputType = m_settings->inputType;

    // If random mix, pick a random type
    if (inputType == 3) {
        std::uniform_int_distribution<int> dist(0, 2);
        inputType = dist(m_rng);
    }

    switch (inputType) {
        case 0:
            SendArrowKeyInput(wowWnd);
            break;
        case 1:
            SendJumpInput(wowWnd);
            break;
        case 2:
            SendMouseWiggle(wowWnd);
            break;
        default:
            SendArrowKeyInput(wowWnd);
            break;
    }
}

void AntiAFK::SendArrowKeyInput(HWND wowWnd) {
    // Send a very brief arrow key tap (left or right randomly)
    // This causes minimal character movement

    std::uniform_int_distribution<int> dist(0, 1);
    WORD key = dist(m_rng) ? VK_LEFT : VK_RIGHT;

    // Random very short duration (10-50ms) - barely noticeable
    std::uniform_int_distribution<int> durationDist(10, 50);
    int duration = durationDist(m_rng);

    // Send key down
    INPUT inputs[2] = {};

    inputs[0].type = INPUT_KEYBOARD;
    inputs[0].ki.wVk = key;
    inputs[0].ki.dwFlags = 0;

    inputs[1].type = INPUT_KEYBOARD;
    inputs[1].ki.wVk = key;
    inputs[1].ki.dwFlags = KEYEVENTF_KEYUP;

    // Bring WoW to foreground briefly if needed, or use PostMessage
    // Using SendInput requires the window to be focused
    HWND currentForeground = GetForegroundWindow();

    if (currentForeground == wowWnd) {
        // WoW is focused, use SendInput
        SendInput(1, &inputs[0], sizeof(INPUT));
        Sleep(duration);
        SendInput(1, &inputs[1], sizeof(INPUT));
    } else {
        // WoW is not focused, use PostMessage (works on background windows)
        PostMessage(wowWnd, WM_KEYDOWN, key, 0);
        Sleep(duration);
        PostMessage(wowWnd, WM_KEYUP, key, 0);
    }
}

void AntiAFK::SendJumpInput(HWND wowWnd) {
    // Send a spacebar press (jump)
    // This is very visible but effective

    std::uniform_int_distribution<int> durationDist(30, 80);
    int duration = durationDist(m_rng);

    INPUT inputs[2] = {};

    inputs[0].type = INPUT_KEYBOARD;
    inputs[0].ki.wVk = VK_SPACE;
    inputs[0].ki.dwFlags = 0;

    inputs[1].type = INPUT_KEYBOARD;
    inputs[1].ki.wVk = VK_SPACE;
    inputs[1].ki.dwFlags = KEYEVENTF_KEYUP;

    HWND currentForeground = GetForegroundWindow();

    if (currentForeground == wowWnd) {
        SendInput(1, &inputs[0], sizeof(INPUT));
        Sleep(duration);
        SendInput(1, &inputs[1], sizeof(INPUT));
    } else {
        PostMessage(wowWnd, WM_KEYDOWN, VK_SPACE, 0);
        Sleep(duration);
        PostMessage(wowWnd, WM_KEYUP, VK_SPACE, 0);
    }
}

void AntiAFK::SendMouseWiggle(HWND wowWnd) {
    // Move mouse slightly within the WoW window
    // This is the least noticeable method

    RECT rect;
    if (!GetWindowRect(wowWnd, &rect)) return;

    // Get current mouse position
    POINT currentPos;
    GetCursorPos(&currentPos);

    // Only wiggle if mouse is within WoW window (to avoid disrupting other work)
    if (currentPos.x >= rect.left && currentPos.x <= rect.right &&
        currentPos.y >= rect.top && currentPos.y <= rect.bottom) {

        std::uniform_int_distribution<int> moveDist(-3, 3);
        int dx = moveDist(m_rng);
        int dy = moveDist(m_rng);

        // Ensure at least some movement
        if (dx == 0 && dy == 0) dx = 1;

        INPUT inputs[2] = {};

        // Move mouse
        inputs[0].type = INPUT_MOUSE;
        inputs[0].mi.dx = dx;
        inputs[0].mi.dy = dy;
        inputs[0].mi.dwFlags = MOUSEEVENTF_MOVE;

        // Move back
        inputs[1].type = INPUT_MOUSE;
        inputs[1].mi.dx = -dx;
        inputs[1].mi.dy = -dy;
        inputs[1].mi.dwFlags = MOUSEEVENTF_MOVE;

        SendInput(1, &inputs[0], sizeof(INPUT));
        Sleep(50);
        SendInput(1, &inputs[1], sizeof(INPUT));
    } else {
        // Mouse not in WoW window, fall back to keyboard input
        SendArrowKeyInput(wowWnd);
    }
}
