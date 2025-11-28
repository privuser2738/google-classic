# NoAFK - World of Warcraft Anti-AFK Tool

A lightweight Windows system tray application that prevents your World of Warcraft character from being disconnected due to inactivity.

## Features

- **System Tray Application**: Runs quietly in the background with a tray icon
- **Randomized Timing**: Sends inputs at random intervals to appear natural
- **Multiple Input Methods**:
  - Arrow keys (minimal movement)
  - Jump (spacebar)
  - Mouse wiggle (least noticeable)
  - Random mix of all methods
- **Configurable Settings**:
  - Adjustable min/max intervals
  - Focus-only mode (only active when WoW is focused)
  - Custom process name detection
- **Visual Feedback**: Green icon when active, orange when paused

## Building

### Requirements

- Windows 10/11
- One of the following:
  - Visual Studio 2019/2022 with C++ workload
  - MinGW-w64 / MSYS2
- CMake 3.16+ (for CMake build)

### Build with CMake (Recommended)

```batch
# Create build directory
mkdir build
cd build

# Configure (Visual Studio)
cmake -G "Visual Studio 17 2022" ..

# Or for MinGW
cmake -G "MinGW Makefiles" ..

# Build
cmake --build . --config Release
```

### Build with Make (MinGW/MSYS2)

```bash
make release
```

### Build Script (Windows)

```batch
build.bat
```

The executable will be created at `build/bin/NoAFK.exe`.

## Usage

1. Run `NoAFK.exe`
2. The application will minimize to the system tray with a notification
3. Right-click the tray icon for options:
   - **Status**: Shows current state and WoW detection
   - **Settings**: Configure timing and input methods
   - **Pause/Resume**: Temporarily disable anti-AFK
   - **Exit**: Close the application

## Settings

| Setting | Description | Default |
|---------|-------------|---------|
| Min Interval | Minimum seconds between inputs | 30 |
| Max Interval | Maximum seconds between inputs | 120 |
| Input Type | Method used to prevent AFK | Arrow Keys |
| Only When Focused | Only send inputs when WoW is active | Off |
| Process Name | WoW executable to detect | Wow.exe |

## How It Works

The application detects the WoW game window using the `GxWindowClass` window class and process name matching. At random intervals within your configured range, it sends a brief input to prevent the AFK timer from triggering.

- **Arrow Keys**: Sends a very brief left/right tap (10-50ms) - causes minimal character movement
- **Jump**: Sends spacebar press - visible but effective
- **Mouse Wiggle**: Moves mouse ±3 pixels and back - only works if mouse is in WoW window
- **Random Mix**: Randomly selects from the above methods

## Supported WoW Versions

- World of Warcraft (Retail)
- WoW Classic
- All regional clients (Wow.exe, WowClassic.exe, Wow-64.exe, WoWT.exe, WoWB.exe)

## License

Free software - use at your own risk.

## Disclaimer

This tool is for personal use to prevent disconnection during legitimate gameplay. Use responsibly and in accordance with Blizzard's Terms of Service.
