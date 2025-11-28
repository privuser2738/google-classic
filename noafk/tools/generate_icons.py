#!/usr/bin/env python3
"""
Generate icons for NoAFK application.
Creates both active and paused state icons.
"""

import struct
import os

def create_ico_file(filename, image_data_16, image_data_32):
    """Create an ICO file with 16x16 and 32x32 images."""

    # ICO header: reserved(2) + type(2) + count(2)
    header = struct.pack('<HHH', 0, 1, 2)

    # Calculate offsets
    icon_dir_size = 6 + (16 * 2)  # header + 2 directory entries
    offset_16 = icon_dir_size
    offset_32 = offset_16 + len(image_data_16)

    # Icon directory entries
    # Width, Height, Colors, Reserved, Planes, BitCount, Size, Offset
    entry_16 = struct.pack('<BBBBHHII', 16, 16, 0, 0, 1, 32, len(image_data_16), offset_16)
    entry_32 = struct.pack('<BBBBHHII', 32, 32, 0, 0, 1, 32, len(image_data_32), offset_32)

    with open(filename, 'wb') as f:
        f.write(header)
        f.write(entry_16)
        f.write(entry_32)
        f.write(image_data_16)
        f.write(image_data_32)

def create_bmp_data(size, color_func):
    """Create BMP image data for an icon."""

    # BITMAPINFOHEADER (40 bytes)
    header = struct.pack('<IIIHHIIIIII',
        40,           # biSize
        size,         # biWidth
        size * 2,     # biHeight (doubled for AND mask)
        1,            # biPlanes
        32,           # biBitCount
        0,            # biCompression
        0,            # biSizeImage
        0,            # biXPelsPerMeter
        0,            # biYPelsPerMeter
        0,            # biClrUsed
        0             # biClrImportant
    )

    # Generate pixel data (BGRA format, bottom-up)
    pixels = bytearray()
    for y in range(size - 1, -1, -1):
        for x in range(size):
            b, g, r, a = color_func(x, y, size)
            pixels.extend([b, g, r, a])

    # AND mask (1 bit per pixel, padded to 4 bytes per row)
    mask_row_size = ((size + 31) // 32) * 4
    mask = bytearray(mask_row_size * size)  # All zeros = fully opaque

    return header + bytes(pixels) + bytes(mask)

def active_icon_color(x, y, size):
    """Generate colors for the active (green) icon."""
    center = size / 2
    radius = size / 2 - 1

    dx = x - center + 0.5
    dy = y - center + 0.5
    dist = (dx * dx + dy * dy) ** 0.5

    if dist <= radius:
        # Inside circle - gradient green
        inner_radius = radius * 0.6
        if dist <= inner_radius:
            # Inner bright green
            intensity = 1.0 - (dist / inner_radius) * 0.3
            r = int(50 * intensity)
            g = int(220 * intensity)
            b = int(80 * intensity)
            return (b, g, r, 255)
        else:
            # Outer darker green ring
            t = (dist - inner_radius) / (radius - inner_radius)
            r = int(30 + (50 - 30) * (1 - t))
            g = int(150 + (220 - 150) * (1 - t))
            b = int(50 + (80 - 50) * (1 - t))

            # Anti-aliasing at edge
            if dist > radius - 1:
                alpha = int(255 * (radius - dist + 1))
                return (b, g, r, max(0, min(255, alpha)))
            return (b, g, r, 255)
    else:
        return (0, 0, 0, 0)  # Transparent

def paused_icon_color(x, y, size):
    """Generate colors for the paused (orange/yellow) icon."""
    center = size / 2
    radius = size / 2 - 1

    dx = x - center + 0.5
    dy = y - center + 0.5
    dist = (dx * dx + dy * dy) ** 0.5

    if dist <= radius:
        # Inside circle - gradient orange/yellow
        inner_radius = radius * 0.6
        if dist <= inner_radius:
            # Inner bright orange
            intensity = 1.0 - (dist / inner_radius) * 0.3
            r = int(255 * intensity)
            g = int(180 * intensity)
            b = int(50 * intensity)
            return (b, g, r, 255)
        else:
            # Outer darker orange ring
            t = (dist - inner_radius) / (radius - inner_radius)
            r = int(200 + (255 - 200) * (1 - t))
            g = int(120 + (180 - 120) * (1 - t))
            b = int(30 + (50 - 30) * (1 - t))

            # Anti-aliasing at edge
            if dist > radius - 1:
                alpha = int(255 * (radius - dist + 1))
                return (b, g, r, max(0, min(255, alpha)))
            return (b, g, r, 255)
    else:
        return (0, 0, 0, 0)  # Transparent

def main():
    # Ensure output directory exists
    icons_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'icons')
    os.makedirs(icons_dir, exist_ok=True)

    # Generate active icon
    active_16 = create_bmp_data(16, active_icon_color)
    active_32 = create_bmp_data(32, active_icon_color)
    create_ico_file(os.path.join(icons_dir, 'noafk.ico'), active_16, active_32)
    print(f"Created: {os.path.join(icons_dir, 'noafk.ico')}")

    # Generate paused icon
    paused_16 = create_bmp_data(16, paused_icon_color)
    paused_32 = create_bmp_data(32, paused_icon_color)
    create_ico_file(os.path.join(icons_dir, 'noafk_paused.ico'), paused_16, paused_32)
    print(f"Created: {os.path.join(icons_dir, 'noafk_paused.ico')}")

if __name__ == '__main__':
    main()
