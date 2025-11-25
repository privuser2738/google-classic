# Malgoro Themes

Visual customization packages for the Malgoro desktop environment.

## About

This repository contains themes, icons, cursors, wallpapers, and sounds that provide the classic 2000s aesthetic that defines Malgoro's visual identity.

## Packages

This repository contains **10 visual customization packages**:

### Themes (5 packages)

| Package | Colors | Style | Inspired By |
|---------|--------|-------|-------------|
| **malgoro-theme-classic** | Gray/Blue | Traditional | Windows 2000 |
| **malgoro-theme-luna** | Blue/Green | Modern classic | Windows XP |
| **malgoro-theme-royale** | Deep Blue | Professional | Windows XP Media Center |
| **malgoro-theme-olive** | Olive/Green | Natural | Windows XP Olive |
| **malgoro-theme-silver** | Gray/Silver | Elegant | Windows XP Silver |

### Icons & Cursors (3 packages)

| Package | Description |
|---------|-------------|
| **malgoro-icons-classic** | Classic icon set with 2000s styling |
| **malgoro-icons-modern** | Modern flat icons with classic colors |
| **malgoro-cursors** | Classic cursor themes |

### Media (2 packages)

| Package | Description |
|---------|-------------|
| **malgoro-wallpapers** | Collection of classic-themed wallpapers |
| **malgoro-sounds** | System sounds and notifications |

## Installation

### Install Default Theme (Luna)

```bash
# Add repository
malgoropkg repo-add malgoro-themes https://github.com/privuser2738/malgoro-themes

# Sync
malgoropkg sync

# Install Luna theme with icons
malgoropkg install malgoro-theme-luna malgoro-icons-classic malgoro-cursors
```

### Install All Themes

```bash
malgoropkg install malgoro-theme-classic malgoro-theme-luna malgoro-theme-royale \
    malgoro-theme-olive malgoro-theme-silver
```

### Install All Visual Packages

```bash
# Everything!
malgoropkg install malgoro-theme-luna malgoro-icons-classic malgoro-icons-modern \
    malgoro-cursors malgoro-wallpapers malgoro-sounds
```

## Theme Details

### Classic Theme
**Colors**: Gray (#D4D0C8), Blue (#0054E3)
**Style**: Traditional Windows 2000 look
**Best For**: Users who prefer simplicity and tradition

### Luna Theme (Default)
**Colors**: Blue (#0054E3), Green (#73D216), Orange (#F57900)
**Style**: The iconic Windows XP look with gradients
**Best For**: Nostalgic users, default choice

### Royale Theme
**Colors**: Deep Blue (#003D79), Navy (#001F3F)
**Style**: Professional Windows XP Media Center style
**Best For**: Professional environments, media enthusiasts

### Olive Theme
**Colors**: Olive Green (#6B8E23), Earth tones
**Style**: Natural, earthy Windows XP variant
**Best For**: Users who prefer green/natural colors

### Silver Theme
**Colors**: Silver (#C0C0C0), Gray tones
**Style**: Elegant, metallic Windows XP look
**Best For**: Modern professional aesthetic

## Switching Themes

### Using Settings GUI

```bash
malgoro-settings
# Navigate to Appearance → Themes
# Select your preferred theme
```

### Using Command Line

```bash
# Set theme
malgoro-settings --set-theme luna

# Set icon theme
malgoro-settings --set-icons classic

# Set wallpaper
malgoro-settings --set-wallpaper /usr/share/backgrounds/malgoro/bliss.jpg
```

## Theme Components

Each theme package includes:

- **Window Manager Theme** - Titlebar colors, borders, buttons
- **GTK Theme** - Application widget styling
- **Metacity/Openbox Theme** - Window decoration files
- **Color Scheme** - System-wide color definitions

## Package Sizes

| Package | Approximate Size |
|---------|------------------|
| Theme packages | ~500KB each |
| Icon packages | ~5-10MB each |
| Cursor package | ~1MB |
| Wallpapers | ~20-30MB |
| Sounds | ~2-3MB |

**Total**: ~50-60MB for all visual packages

## Creating Custom Themes

See the [Theme Creation Guide](https://github.com/privuser2738/malgoro/docs/THEME_CREATION.md) for instructions on creating your own themes.

## Screenshots

### Luna Theme
*The classic Windows XP look*

### Royale Theme
*Professional deep blue aesthetic*

### Classic Theme
*Traditional Windows 2000 style*

## Dependencies

Most theme packages have no runtime dependencies, but some require:
- GTK3 theme engine
- Icon theme engine
- X11 cursor support

## Contributing

Want to create a theme? Fork this repository and submit a PR!

See [CONTRIBUTING.md](https://github.com/privuser2738/malgoro/blob/main/CONTRIBUTING.md)

## License

MIT License - See [LICENSE](LICENSE) file.

Individual wallpapers and icons may have different licenses (CC-BY, CC0, etc.) - see package-specific LICENSE files.

## Links

- **Main Repository**: https://github.com/privuser2738/malgoro
- **Core Packages**: https://github.com/privuser2738/malgoro-core
- **Issue Tracker**: https://github.com/privuser2738/malgoro/issues

---

**Part of the Malgoro Desktop Environment**

*Bringing back the elegance of classic desktop computing*
