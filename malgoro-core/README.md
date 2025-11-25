# Malgoro Core Packages

Core desktop environment components for Malgoro.

## About

This repository contains the essential packages that make up the Malgoro desktop environment. These packages work together to provide a complete, classic 2000s-inspired desktop experience.

## Packages

This repository contains **5 core packages**:

| Package | Version | Description |
|---------|---------|-------------|
| **malgoro-wm** | 0.1.0 | Window manager with classic decorations |
| **malgoro-panel** | 0.1.0 | Desktop panel with taskbar and system tray |
| **malgoro-launcher** | 0.1.0 | Application launcher with search |
| **malgoro-settings** | 0.1.0 | Settings and configuration manager |
| **malgoro-session** | 0.1.0 | Session manager and startup |

## Installation

### Install All Core Packages

```bash
# Add repository
malgoropkg repo-add malgoro-core https://github.com/privuser2738/malgoro-core

# Sync
malgoropkg sync

# Install all core packages
malgoropkg install malgoro-wm malgoro-panel malgoro-launcher malgoro-settings malgoro-session
```

### Install Individual Packages

```bash
# Just the window manager
malgoropkg install malgoro-wm

# Window manager + panel
malgoropkg install malgoro-wm malgoro-panel
```

## Package Details

### malgoro-wm (Window Manager)

The core window manager providing:
- Classic window decorations (24px titlebars, 1px borders)
- Window management (move, resize, minimize, maximize, close)
- Workspace support
- Keyboard shortcuts (Alt+Tab, Alt+F4, etc.)
- ICCCM and EWMH compliance

**Dependencies**: libx11, libxft, libwnck-3, cairo, pango

### malgoro-panel (Desktop Panel)

Traditional desktop panel featuring:
- Taskbar showing open windows
- System tray for background applications
- Clock and calendar
- Application menu button
- Workspace switcher
- Configurable position (top/bottom/left/right)

**Dependencies**: gtk3, libwnck-3, cairo, dbus

### malgoro-launcher (Application Launcher)

Quick application launcher with:
- Search functionality
- Recent applications
- Favorites
- Category browsing
- Desktop file integration

**Dependencies**: gtk3, glib2

### malgoro-settings (Settings Manager)

Centralized settings for:
- Theme selection
- Keyboard shortcuts
- Display configuration
- Panel preferences
- Window manager settings
- Startup applications

**Dependencies**: gtk3, dbus

### malgoro-session (Session Manager)

Session management providing:
- Desktop session startup
- Component launching (WM, panel, etc.)
- Session saving and restoration
- Logout/reboot/shutdown dialogs
- Auto-start application handling

**Dependencies**: dbus, systemd (optional)

## Building from Source

Each package can be built independently:

```bash
# Clone repository
git clone https://github.com/privuser2738/malgoro-core
cd malgoro-core

# Build specific package
cd packages/malgoro-wm
./build.sh

# Or build all packages
for pkg in packages/*/; do
    cd "$pkg"
    ./build.sh
    cd ../..
done
```

## Development Status

| Package | Status | Progress |
|---------|--------|----------|
| malgoro-wm | 🔨 In Development | Architecture complete, implementation in progress |
| malgoro-panel | 📋 Planned | Architecture complete |
| malgoro-launcher | 📋 Planned | Architecture complete |
| malgoro-settings | 📋 Planned | Design phase |
| malgoro-session | 📋 Planned | Design phase |

## Contributing

See the main [Malgoro repository](https://github.com/privuser2738/malgoro) for contribution guidelines.

## Dependencies

All packages require:
- **Build**: cmake >= 3.20, gcc >= 11 or clang >= 14
- **Runtime**: X11, GTK3

Specific dependencies are listed in each package's PKGINFO file.

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Links

- **Main Repository**: https://github.com/privuser2738/malgoro
- **Documentation**: https://github.com/privuser2738/malgoro/tree/main/docs
- **Issue Tracker**: https://github.com/privuser2738/malgoro/issues

---

**Part of the Malgoro Desktop Environment**

*Bringing back the elegance of classic desktop computing*
