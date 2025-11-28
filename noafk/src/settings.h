/*
 * NoAFK - Settings Header
 * Configuration management for anti-AFK behavior
 */

#ifndef SETTINGS_H
#define SETTINGS_H

#define WIN32_LEAN_AND_MEAN
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif

#include <windows.h>
#include <string>

// Default settings
#define DEFAULT_MIN_INTERVAL    30000   // 30 seconds
#define DEFAULT_MAX_INTERVAL    120000  // 2 minutes
#define DEFAULT_INPUT_TYPE      0       // 0 = Movement keys, 1 = Jump, 2 = Mouse move
#define DEFAULT_ONLY_WHEN_FOCUS false
#define DEFAULT_START_MINIMIZED true

class Settings {
public:
    // Timing settings (in milliseconds)
    DWORD minInterval = DEFAULT_MIN_INTERVAL;
    DWORD maxInterval = DEFAULT_MAX_INTERVAL;

    // Input type: 0 = Arrow keys, 1 = Jump (Space), 2 = Mouse wiggle, 3 = Random mix
    int inputType = DEFAULT_INPUT_TYPE;

    // Only send inputs when WoW is focused
    bool onlyWhenFocused = DEFAULT_ONLY_WHEN_FOCUS;

    // Start minimized to tray
    bool startMinimized = DEFAULT_START_MINIMIZED;

    // Target process name
    std::wstring processName = L"Wow.exe";

    // Load settings from registry
    void Load();

    // Save settings to registry
    void Save();

    // Reset to defaults
    void Reset();
};

// Show the settings dialog
// Returns true if settings were changed
bool ShowSettingsDialog(HINSTANCE hInstance, HWND hParent, Settings* settings);

#endif // SETTINGS_H
