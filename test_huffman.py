import unittest
from huffman import HPixel
from PIL import Image
import numpy as np

class TestHuffmanKeyConversion(unittest.TestCase):
    def setUp(self):
        """Set up common resources for all tests."""
        self.img_path = "/home/naughtius-maximus/Desktop/ID/dp.bmp"
        self.encoder = HPixel(self.img_path)
        self.decoder = HPixel(self.img_path.replace(".bmp", ".json"))

    def test_key_to_int_and_int_to_key(self):
        # Test cases for RGB tuples
        test_cases = [
            (0, 0, 0),
            (1, 256, 256)
        ]

        for rgb in test_cases:
            with self.subTest(rgb=rgb):
                rgb_int = self.encoder.key_to_int(rgb)
                rgb_back = self.encoder.int_to_key(rgb_int)
                self.assertEqual(rgb, rgb_back)

    def test_transform_and_itransform_pixels(self):
        # Test cases for pixel transformations
        test_cases = [
            ([(1, 1, 1), (2, 2, 2), (2, 2, 2), (4, 4, 4)], (2, 2))
        ]

        for pixels, size in test_cases:
            with self.subTest(pixels=pixels, size=size):
                transformed = self.encoder.transform_pixels(pixels, size)
                restored = self.encoder.itransform_pixels(transformed, pixels[0], size)
                self.assertEqual(restored, pixels)

    def test_image_integrity_after_encoding_decoding(self):
        # Load the original and decoded images
        original_image = np.array(Image.open(self.img_path))
        decoded_image_array = np.array(Image.open(self.img_path.replace(".bmp", "_1.bmp")))

        # Assert that the images are the same
        self.assertTrue(np.array_equal(original_image, decoded_image_array), "The decoded image is not the same as the original.")

if __name__ == "__main__":
    unittest.main()