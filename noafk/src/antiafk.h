/*
 * NoAFK - Anti-AFK Engine Header
 * Core logic for preventing AFK timeout
 */

#ifndef ANTIAFK_H
#define ANTIAFK_H

#define WIN32_LEAN_AND_MEAN
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif

#include <windows.h>
#include <tlhelp32.h>
#include <random>
#include <atomic>
#include <thread>

#include "settings.h"

class AntiAFK {
public:
    AntiAFK();
    ~AntiAFK();

    // Set configuration
    void SetSettings(Settings* settings);

    // Start the anti-AFK background thread
    void Start(HWND notifyWnd);

    // Stop the anti-AFK thread
    void Stop();

    // Pause without stopping thread
    void Pause();

    // Resume after pause
    void Resume();

    // Check if WoW is currently running
    bool IsWoWRunning();

    // Check if WoW window is focused
    bool IsWoWFocused();

    // Get time until next action (ms)
    DWORD GetTimeUntilNext() const;

private:
    Settings* m_settings = nullptr;
    HWND m_notifyWnd = nullptr;

    std::atomic<bool> m_running{false};
    std::atomic<bool> m_paused{false};
    std::thread m_thread;

    std::mt19937 m_rng;
    DWORD m_nextActionTime = 0;

    // Worker thread function
    void WorkerThread();

    // Find WoW process
    HWND FindWoWWindow();

    // Send a dummy input to prevent AFK
    void SendAntiAFKInput(HWND wowWnd);

    // Different input methods
    void SendArrowKeyInput(HWND wowWnd);
    void SendJumpInput(HWND wowWnd);
    void SendMouseWiggle(HWND wowWnd);

    // Get random interval within configured range
    DWORD GetRandomInterval();
};

#endif // ANTIAFK_H
