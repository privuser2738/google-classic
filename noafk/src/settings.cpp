/*
 * NoAFK - Settings Implementation
 * Configuration management and settings dialog
 */

#include "settings.h"
#include "resource.h"
#include <commctrl.h>
#include <shlobj.h>
#include <sstream>

#ifdef _MSC_VER
#pragma comment(lib, "comctl32.lib")
#endif

static const wchar_t* REG_KEY = L"Software\\NoAFK";

void Settings::Load() {
    HKEY hKey;
    if (RegOpenKeyEx(HKEY_CURRENT_USER, REG_KEY, 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        DWORD size = sizeof(DWORD);
        DWORD value;

        if (RegQueryValueEx(hKey, L"MinInterval", nullptr, nullptr, (LPBYTE)&value, &size) == ERROR_SUCCESS) {
            minInterval = value;
        }
        if (RegQueryValueEx(hKey, L"MaxInterval", nullptr, nullptr, (LPBYTE)&value, &size) == ERROR_SUCCESS) {
            maxInterval = value;
        }
        if (RegQueryValueEx(hKey, L"InputType", nullptr, nullptr, (LPBYTE)&value, &size) == ERROR_SUCCESS) {
            inputType = (int)value;
        }
        if (RegQueryValueEx(hKey, L"OnlyWhenFocused", nullptr, nullptr, (LPBYTE)&value, &size) == ERROR_SUCCESS) {
            onlyWhenFocused = (value != 0);
        }
        if (RegQueryValueEx(hKey, L"StartMinimized", nullptr, nullptr, (LPBYTE)&value, &size) == ERROR_SUCCESS) {
            startMinimized = (value != 0);
        }

        wchar_t buffer[256];
        size = sizeof(buffer);
        if (RegQueryValueEx(hKey, L"ProcessName", nullptr, nullptr, (LPBYTE)buffer, &size) == ERROR_SUCCESS) {
            processName = buffer;
        }

        RegCloseKey(hKey);
    }

    // Validate
    if (minInterval < 5000) minInterval = 5000;
    if (maxInterval < minInterval) maxInterval = minInterval + 30000;
    if (maxInterval > 300000) maxInterval = 300000;
}

void Settings::Save() {
    HKEY hKey;
    if (RegCreateKeyEx(HKEY_CURRENT_USER, REG_KEY, 0, nullptr, 0, KEY_WRITE, nullptr, &hKey, nullptr) == ERROR_SUCCESS) {
        RegSetValueEx(hKey, L"MinInterval", 0, REG_DWORD, (LPBYTE)&minInterval, sizeof(DWORD));
        RegSetValueEx(hKey, L"MaxInterval", 0, REG_DWORD, (LPBYTE)&maxInterval, sizeof(DWORD));

        DWORD value = (DWORD)inputType;
        RegSetValueEx(hKey, L"InputType", 0, REG_DWORD, (LPBYTE)&value, sizeof(DWORD));

        value = onlyWhenFocused ? 1 : 0;
        RegSetValueEx(hKey, L"OnlyWhenFocused", 0, REG_DWORD, (LPBYTE)&value, sizeof(DWORD));

        value = startMinimized ? 1 : 0;
        RegSetValueEx(hKey, L"StartMinimized", 0, REG_DWORD, (LPBYTE)&value, sizeof(DWORD));

        RegSetValueEx(hKey, L"ProcessName", 0, REG_SZ, (LPBYTE)processName.c_str(),
                      (DWORD)((processName.length() + 1) * sizeof(wchar_t)));

        RegCloseKey(hKey);
    }
}

void Settings::Reset() {
    minInterval = DEFAULT_MIN_INTERVAL;
    maxInterval = DEFAULT_MAX_INTERVAL;
    inputType = DEFAULT_INPUT_TYPE;
    onlyWhenFocused = DEFAULT_ONLY_WHEN_FOCUS;
    startMinimized = DEFAULT_START_MINIMIZED;
    processName = L"Wow.exe";
}

// Dialog data
struct SettingsDialogData {
    Settings* settings;
    bool changed;
};

INT_PTR CALLBACK SettingsDialogProc(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam) {
    SettingsDialogData* data = (SettingsDialogData*)GetWindowLongPtr(hDlg, GWLP_USERDATA);

    switch (message) {
        case WM_INITDIALOG: {
            data = (SettingsDialogData*)lParam;
            SetWindowLongPtr(hDlg, GWLP_USERDATA, (LONG_PTR)data);

            // Set window icon
            HICON hIcon = LoadIcon(GetModuleHandle(nullptr), MAKEINTRESOURCE(IDI_APPICON));
            SendMessage(hDlg, WM_SETICON, ICON_SMALL, (LPARAM)hIcon);
            SendMessage(hDlg, WM_SETICON, ICON_BIG, (LPARAM)hIcon);

            // Initialize controls with current values
            SetDlgItemInt(hDlg, IDC_MIN_INTERVAL, data->settings->minInterval / 1000, FALSE);
            SetDlgItemInt(hDlg, IDC_MAX_INTERVAL, data->settings->maxInterval / 1000, FALSE);

            // Set up combo box for input type
            HWND hCombo = GetDlgItem(hDlg, IDC_INPUT_TYPE);
            SendMessage(hCombo, CB_ADDSTRING, 0, (LPARAM)L"Arrow Keys (tap left/right)");
            SendMessage(hCombo, CB_ADDSTRING, 0, (LPARAM)L"Jump (spacebar)");
            SendMessage(hCombo, CB_ADDSTRING, 0, (LPARAM)L"Mouse Wiggle");
            SendMessage(hCombo, CB_ADDSTRING, 0, (LPARAM)L"Random Mix");
            SendMessage(hCombo, CB_SETCURSEL, data->settings->inputType, 0);

            // Checkboxes
            CheckDlgButton(hDlg, IDC_ONLY_FOCUSED, data->settings->onlyWhenFocused ? BST_CHECKED : BST_UNCHECKED);
            CheckDlgButton(hDlg, IDC_START_MINIMIZED, data->settings->startMinimized ? BST_CHECKED : BST_UNCHECKED);

            // Process name
            SetDlgItemText(hDlg, IDC_PROCESS_NAME, data->settings->processName.c_str());

            // Center dialog on screen
            RECT rc;
            GetWindowRect(hDlg, &rc);
            int x = (GetSystemMetrics(SM_CXSCREEN) - (rc.right - rc.left)) / 2;
            int y = (GetSystemMetrics(SM_CYSCREEN) - (rc.bottom - rc.top)) / 2;
            SetWindowPos(hDlg, nullptr, x, y, 0, 0, SWP_NOSIZE | SWP_NOZORDER);

            return TRUE;
        }

        case WM_COMMAND:
            switch (LOWORD(wParam)) {
                case IDOK: {
                    // Retrieve and validate values
                    DWORD minInt = GetDlgItemInt(hDlg, IDC_MIN_INTERVAL, nullptr, FALSE) * 1000;
                    DWORD maxInt = GetDlgItemInt(hDlg, IDC_MAX_INTERVAL, nullptr, FALSE) * 1000;

                    if (minInt < 5000) {
                        MessageBox(hDlg, L"Minimum interval must be at least 5 seconds.", L"Invalid Setting", MB_ICONWARNING);
                        return TRUE;
                    }
                    if (maxInt < minInt) {
                        MessageBox(hDlg, L"Maximum interval must be greater than minimum.", L"Invalid Setting", MB_ICONWARNING);
                        return TRUE;
                    }
                    if (maxInt > 300000) {
                        MessageBox(hDlg, L"Maximum interval cannot exceed 5 minutes.", L"Invalid Setting", MB_ICONWARNING);
                        return TRUE;
                    }

                    // Save values
                    data->settings->minInterval = minInt;
                    data->settings->maxInterval = maxInt;
                    data->settings->inputType = (int)SendDlgItemMessage(hDlg, IDC_INPUT_TYPE, CB_GETCURSEL, 0, 0);
                    data->settings->onlyWhenFocused = (IsDlgButtonChecked(hDlg, IDC_ONLY_FOCUSED) == BST_CHECKED);
                    data->settings->startMinimized = (IsDlgButtonChecked(hDlg, IDC_START_MINIMIZED) == BST_CHECKED);

                    wchar_t buffer[256];
                    GetDlgItemText(hDlg, IDC_PROCESS_NAME, buffer, 256);
                    data->settings->processName = buffer;

                    data->settings->Save();
                    data->changed = true;

                    EndDialog(hDlg, IDOK);
                    return TRUE;
                }

                case IDCANCEL:
                    EndDialog(hDlg, IDCANCEL);
                    return TRUE;

                case IDC_RESET:
                    data->settings->Reset();
                    // Update controls
                    SetDlgItemInt(hDlg, IDC_MIN_INTERVAL, data->settings->minInterval / 1000, FALSE);
                    SetDlgItemInt(hDlg, IDC_MAX_INTERVAL, data->settings->maxInterval / 1000, FALSE);
                    SendDlgItemMessage(hDlg, IDC_INPUT_TYPE, CB_SETCURSEL, data->settings->inputType, 0);
                    CheckDlgButton(hDlg, IDC_ONLY_FOCUSED, data->settings->onlyWhenFocused ? BST_CHECKED : BST_UNCHECKED);
                    CheckDlgButton(hDlg, IDC_START_MINIMIZED, data->settings->startMinimized ? BST_CHECKED : BST_UNCHECKED);
                    SetDlgItemText(hDlg, IDC_PROCESS_NAME, data->settings->processName.c_str());
                    return TRUE;
            }
            break;

        case WM_CLOSE:
            EndDialog(hDlg, IDCANCEL);
            return TRUE;
    }
    return FALSE;
}

bool ShowSettingsDialog(HINSTANCE hInstance, HWND hParent, Settings* settings) {
    SettingsDialogData data;
    data.settings = settings;
    data.changed = false;

    DialogBoxParam(hInstance, MAKEINTRESOURCE(IDD_SETTINGS), hParent, SettingsDialogProc, (LPARAM)&data);

    return data.changed;
}
