import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from vtf import _decode_dxt


class DxtColorPaletteTests(unittest.TestCase):
    def setUp(self):
        # Endpoint 0 is numerically less than endpoint 1. BC1 uses a
        # transparent fourth entry here, whereas BC2 and BC3 must not.
        self.colors = struct.pack("<HHI", 0x001F, 0xF800, 0xEEEEEEEE)

    def test_dxt1_retains_transparent_palette_entry(self):
        rgba = _decode_dxt(self.colors, 4, 4, "DXT1_ONEBITALPHA")
        self.assertEqual(tuple(rgba[:4]), (127, 0, 127, 255))
        self.assertEqual(tuple(rgba[4:8]), (0, 0, 0, 0))

    def test_dxt3_uses_four_rgb_colors_and_explicit_alpha(self):
        rgba = _decode_dxt(bytes([0xFF] * 8) + self.colors, 4, 4, "DXT3")
        self.assertEqual(tuple(rgba[:4]), (85, 0, 170, 255))
        self.assertEqual(tuple(rgba[4:8]), (170, 0, 85, 255))

    def test_dxt5_uses_four_rgb_colors_and_separate_alpha(self):
        alpha = bytes((128, 0)) + bytes(6)
        rgba = _decode_dxt(alpha + self.colors, 4, 4, "DXT5")
        self.assertEqual(tuple(rgba[:4]), (85, 0, 170, 128))
        self.assertEqual(tuple(rgba[4:8]), (170, 0, 85, 128))


if __name__ == "__main__":
    unittest.main()
