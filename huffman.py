import os
from abc import ABC, abstractmethod
from collections import Counter
from math import log2, sqrt
import heapq
from PIL import Image
import matplotlib.pyplot as plt
import json
import base64
from collections import defaultdict
import numpy as np
import time

def to_ascii(s):
    # Make sure length is multiple of 8
    padding = (8 - len(s) % 8) % 8
    s += '0' * padding

    # Break into 8-bit chunks and convert to characters
    chars = [
        chr(int(s[i:i+8], 2))
        for i in range(0, len(s), 8)
    ]

    # Join characters into a string
    return ''.join(chars)

def from_ascii(s, size):
    # Convert each character to 8-bit binary
    bits = ''.join(f'{ord(c):08b}' for c in s)
    # Trim padding to restore original bit length
    return bits[:size]


def ito_bytes(l, isize):
    bits = ''.join(f'{x:0{isize}b}' for x in l)
    return to_bytes(bits)    


def to_bytes(s):
    byte_array = int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')
    b64_data = base64.b64encode(byte_array).decode('ascii')
    return b64_data


def from_bytes(s, size):
    return bin(int.from_bytes(base64.b64decode(s), byteorder='big'))[2:].zfill(size)


def bytes_to_int_list(s, size, isize):
    bits = from_bytes(s, size)
    return [
        int(bits[i:i+isize], 2)
        for i in range(0, size, isize)
    ]


class Node:
    def __init__(self, freq, color=None, left=None, right=None):
        self.freq = freq
        self.color = color
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


class HBase(ABC):
    def __init__(self, image_path=None, **kwargs):
        self.image_path = image_path
        if image_path.endswith('.bmp'):
           # encoder
            self.image = Image.open(image_path).convert("RGB")
            start_time = time.time()
            self.pixels = self.image.getdata()
            self.size = self.image.size
            self.tpixels = self.transform_pixels(self.pixels, self.size)
            self.compute_frequencies()  # sets self.freq_map
            root = self.build_huffman_tree()
            self.generate_codes(root)  # sets self.codebook
            self.encode()
            end_time = time.time()
            self.compression_time = end_time - start_time
        else:
            # decoder
            self.image = self.decode()

    @abstractmethod
    def transform_pixels(self, pixels, size):
        pass

    def compute_frequencies(self):
        freq_map = Counter()
        for tpixel in self.tpixels:
            freq_map[tpixel] += 1
        self.freq_map = freq_map

    def _encode(self):
        encoded = []
        for tpixel in self.tpixels:
            encoded.append(self.codebook[tpixel])
        return ''.join(encoded)

    def _decode(self, bits, lookup_table):
        decoded_deltas = []
        current = ""
        count = 0
        bitsize = 0
        for bit in bits:
            current += bit
            bitsize += 1
            if bitsize in lookup_table:
                if current in lookup_table[bitsize]:
                    decoded_deltas.append(lookup_table[bitsize][current])
                    current = ""
                    bitsize = 0
            count += 1
        return decoded_deltas

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass

    def build_huffman_tree(self):
        heap = [Node(freq, symbol) for symbol, freq in self.freq_map.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            a = heapq.heappop(heap)
            b = heapq.heappop(heap)
            heapq.heappush(heap, Node(a.freq + b.freq, None, a, b))
        return heap[0] if heap else None

    def _generate_codes(self, node, prefix="", codebook=None):
        if codebook is None:
            codebook = {}
        if node is None:
            return
        if node.color is not None:
            codebook[node.color] = prefix
        self._generate_codes(node.left, prefix + "0", codebook)
        self._generate_codes(node.right, prefix + "1", codebook)
        return codebook
    
    def generate_codes(self, node):
        codebook = self._generate_codes(node)
        self.codebook = codebook

    @abstractmethod
    def key_to_int(self, rgb):
        pass

    @abstractmethod
    def int_to_key(self, val):
        pass

    def compress_codebook(self):
        codebook = {
            self.key_to_int(k): v for k, v in self.codebook.items()
        }

        _lookup_table = defaultdict(dict)
        for k, v in codebook.items():
            _lookup_table[len(v)][k] = v

        lookup_table = {}
        keys = []
        for l in _lookup_table.keys():
            keys.extend(list(_lookup_table[l].keys()))
            bin_code_per_length = ''.join(_lookup_table[l].values())
            size_code_per_length = len(bin_code_per_length)
            bin_code_per_length = to_bytes(bin_code_per_length)
            lookup_table[l] = [bin_code_per_length, size_code_per_length]

        size_pixel_keys = self.isize * len(keys)
        keys = ito_bytes(keys, self.isize)
        return {
            'keys': [keys, size_pixel_keys],
            'code_values': lookup_table
        }

    def decompress_codebook(self, compressed):
        pixel_data, bit_length = compressed['keys']
        keys = bytes_to_int_list(pixel_data, bit_length, self.isize)
        reverse_codebook = defaultdict(dict)
        index = 0
        
        for length, (ascii_bits, bit_count) in compressed['code_values'].items():
            length = int(length)
            bits = from_bytes(ascii_bits, bit_count)
            codes = [bits[i:i+length] for i in range(0, bit_count, length)]
            for code in codes:
                rgb = self.int_to_key(keys[index])
                reverse_codebook[length][code] = rgb
                index += 1

        return reverse_codebook

    def evaluate(self):
        H = 0
        El = 0
        total = sum(v for v in self.freq_map.values())
        for symbol in self.codebook:
            p = self.freq_map[symbol] / total
            H -= p * log2(p)
            El += p * len(self.codebook[symbol])
        return H, El

    def plot_freq(self):
        counts = [v for v in freq_hist.values()]
        total = sum(counts)
        counts = [v/total for v in counts]

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(counts)), counts, alpha=0.5, label='p(x)')
        plt.bar(range(len(counts)), [-c * log2(c) for c in counts], alpha=0.5, label='El(x)')
        plt.xlabel("Range")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()


class HPixel(HBase):
    def __init__(self, *args, **kwargs):
        self.isize = 8
        super().__init__(*args, **kwargs)

    def transform_pixels(self, pixels, size):
        nums = np.array(pixels).reshape(size[0]*size[1]*3).tolist()
        return nums

    def itransform_pixels(self, decoded_nums):
        decoded_pixels = np.array(decoded_nums).reshape(-1, 3)
        decoded_pixels = list(map(tuple, decoded_pixels))
        return decoded_pixels

    def key_to_int(self, num):
        return num

    def int_to_key(self, num):
        return num
    
    def encode(self):
        code = self._encode()
        encoded_data = to_bytes(code)
        
        codebook = self.compress_codebook()
        json_obj = {
            "size": self.image.size,
            "bit_length": len(code),
            "codebook": codebook,
            "bitstream": encoded_data
        }

        img_ext = self.image_path.split('.')[-1]
        self.json_path = self.image_path.replace(f'.{img_ext}', '.json')
        with open(self.json_path, 'w') as f:
            json.dump(json_obj, f, separators=(',', ':'))  # compact JSON
    
    def decode(self):
        """Decode the image data from a compressed JSON Huffman file."""
        with open(self.image_path, 'r') as f:
            data = json.load(f)

        self.data = data
        self.size = data["size"]
        self.codebook = data["codebook"]
        code_bytes = data["bitstream"]
        self.bits = from_bytes(code_bytes, data["bit_length"])

        lookup_table = self.decompress_codebook(self.codebook)
        decoded_deltas = self._decode(self.bits, lookup_table)
        decoded_pixels = self.itransform_pixels(decoded_deltas)

        img = Image.new("RGB", self.size)
        img.putdata(decoded_pixels)
        img.save(self.image_path.replace(".json", "_1.bmp"))
        return img


class HDelta(HBase):
    def __init__(self, *args, **kwargs):
        self.isize = 27
        # extracts pixels from image
        # implement encode(), decode()
        super().__init__(*args, **kwargs)

    def transform_pixels(self, pixels, size):
        pixels = np.array(pixels).reshape(size[0], size[1], 3)

        delta_left = np.diff(pixels, axis=1)
        delta_top = np.diff(pixels[:, 0, :], axis=0)

        delta_top = np.insert(delta_top, 0, pixels[0, 0, :], axis=0)
        delta = np.insert(delta_left, 0, delta_top, axis=1)

        tpixels = list(map(tuple, delta.reshape(-1, 3)[1:, :]))
        return tpixels

    def itransform_pixels(self, decoded_deltas, initial_pixel, size):
        decoded_pixels = [initial_pixel] + decoded_deltas
        decoded_pixels = np.array(decoded_pixels).reshape(size[0], size[1], 3)
        decoded_pixels[:, 0, :] = np.cumsum(decoded_pixels[:, 0, :], axis=0)
        decoded_pixels = np.cumsum(decoded_pixels, axis=1)
        decoded_pixels = decoded_pixels.reshape(-1, 3)
        decoded_pixels = list(map(tuple, decoded_pixels))
        return decoded_pixels

    def key_to_int(self, rgb):
        """Map RGB tuple from [-255, 255] to a 27-bit integer."""
        r, g, b = [v + 255 for v in rgb]  # map -255..255 → 0..510
        return (r << 18) | (g << 9) | b

    def int_to_key(self, val):
        """Decode 27-bit int back to RGB tuple in range [-255, 255]."""
        r = ((val >> 18) & 0x1FF) - 255
        g = ((val >> 9) & 0x1FF) - 255
        b = (val & 0x1FF) - 255
        return (r, g, b)

    def encode(self):
        code = self._encode()
        encoded_data = to_bytes(code)
        
        codebook = self.compress_codebook()
        json_obj = {
            "size": self.image.size,
            "initial_pixel": self.key_to_int(self.pixels[0]),  # still stored as tuple
            "bit_length": len(code),
            "codebook": codebook,
            "bitstream": encoded_data
        }

        img_ext = self.image_path.split('.')[-1]
        self.json_path = self.image_path.replace(f'.{img_ext}', '.json')
        with open(self.json_path, 'w') as f:
            json.dump(json_obj, f, separators=(',', ':'))  # compact JSON
    
    def decode(self):
        """Decode the image data from a compressed JSON Huffman file."""
        with open(self.image_path, 'r') as f:
            data = json.load(f)

        self.data = data
        self.size = data["size"]
        self.codebook = data["codebook"]
        code_bytes = data["bitstream"]
        self.bits = from_bytes(code_bytes, data["bit_length"])

        self.initial_pixel = self.int_to_key(self.data["initial_pixel"])

        lookup_table = self.decompress_codebook(self.codebook)
        decoded_deltas = self._decode(self.bits, lookup_table)
        decoded_pixels = self.itransform_pixels(decoded_deltas, self.initial_pixel, self.size)

        img = Image.new("RGB", self.size)
        img.putdata(decoded_pixels)
        img.save(self.image_path.replace(".json", "_1.bmp"))
        return img


class HDeltaDouble(HBase):
    def __init__(self, *args, **kwargs):
        self.n = kwargs['n'] if 'n' in kwargs else 2
        self.isize = 27 * self.n
        # extracts pixels from image
        # implement encode(), decode()
        super().__init__(*args, **kwargs)

    def transform_pixels(self, pixels, size):
        n = self.n
        pixels = np.array(pixels).reshape(size[0], size[1], 3)

        delta_left = np.diff(pixels, axis=1)
        delta_top = np.diff(pixels[:, 0, :], axis=0)

        delta_top = np.insert(delta_top, 0, pixels[0, 0, :], axis=0)
        delta = np.insert(delta_left, 0, delta_top, axis=1)

        tpixels = list(map(tuple, delta.reshape(-1, 3)))
        tpixels = [
            tuple(tpixels[i:i+n]) for i in range(0, len(tpixels), n)
        ]
        return tpixels

    def itransform_pixels(self, decoded_deltas, size):
        decoded_pixels = []
        n = self.n
        for i in range(len(decoded_deltas)):
            for j in range(n):
                decoded_pixels.append(decoded_deltas[i][j])
        decoded_pixels = np.array(decoded_pixels).reshape(size[0], size[1], 3)
        decoded_pixels[:, 0, :] = np.cumsum(decoded_pixels[:, 0, :], axis=0)
        decoded_pixels = np.cumsum(decoded_pixels, axis=1)
        decoded_pixels = decoded_pixels.reshape(-1, 3)
        decoded_pixels = list(map(tuple, decoded_pixels))
        return decoded_pixels

    def _key_to_int(self, rgb):
        r, g, b = [v + 255 for v in rgb]  # map -255..255 → 0..510
        return (r << 18) | (g << 9) | b

    def key_to_int(self, rgb):
        """Map RGB tuple from [-255, 255] to a 27-bit integer."""
        num = 0
        n = self.n
        for j in range(n):
            num_ = self._key_to_int(rgb[j])
            num = (num << 27) | num_
        return num

    def _int_to_key(self, val):
        b = (val & 0x1FF) - 255
        g = ((val >> 9) & 0x1FF) - 255
        r = ((val >> 18) & 0x1FF) - 255
        return (r, g, b)

    def int_to_key(self, val):
        """Decode n*27-bit int back to a tuple of n RGB tuples in range [-255, 255]."""
        rgb_list = []
        n = self.n
        for _ in range(n):
            r, g, b = self._int_to_key(val)
            rgb_list.insert(0, (r, g, b))  # Insert at the beginning to reverse the order
            val >>= 27  # Shift by 27 bits to process the next RGB tuple
        return tuple(rgb_list)

    def encode(self):
        code = self._encode()
        encoded_data = to_bytes(code)
        
        codebook = self.compress_codebook()
        json_obj = {
            "size": self.image.size,
            "bit_length": len(code),
            "codebook": codebook,
            "bitstream": encoded_data
        }

        img_ext = self.image_path.split('.')[-1]
        self.json_path = self.image_path.replace(f'.{img_ext}', '.json')
        with open(self.json_path, 'w') as f:
            json.dump(json_obj, f, separators=(',', ':'))  # compact JSON
    
    def decode(self):
        """Decode the image data from a compressed JSON Huffman file."""
        with open(self.image_path, 'r') as f:
            data = json.load(f)

        self.data = data
        self.size = data["size"]
        self.codebook = data["codebook"]
        code_bytes = data["bitstream"]
        self.bits = from_bytes(code_bytes, data["bit_length"])

        lookup_table = self.decompress_codebook(self.codebook)
        decoded_deltas = self._decode(self.bits, lookup_table)
        decoded_pixels = self.itransform_pixels(decoded_deltas, self.size)

        img = Image.new("RGB", self.size)
        img.putdata(decoded_pixels)
        img.save(self.image_path.replace(".json", "_1.bmp"))
        return img


class HDeltaReduce(HBase):
    def __init__(self, *args, **kwargs):
        self.isize = 27
        # extracts pixels from image
        # implement encode(), decode()
        super().__init__(*args, **kwargs)

    def transform_pixels(self, pixels, size):
        pixels = np.array(pixels).reshape(size[0], size[1], 3)

        delta_left = np.diff(pixels, axis=1)
        delta_top = np.diff(pixels[:, 0, :], axis=0)

        delta_top = np.insert(delta_top, 0, pixels[0, 0, :], axis=0)
        delta = np.insert(delta_left, 0, delta_top, axis=1)

        pixels = list(map(tuple, delta.reshape(-1, 3)[1:, :]))

        tpixels = []
        repeat_count = 0
        for i in range(len(pixels)):
            if pixels[i] == pixels[i-1]:
                repeat_count += 1
                if repeat_count > 1 and repeat_count < 256:
                    tpixels[-1] = (repeat_count, 256, 256)
                elif repeat_count == 256:
                    repeat_count = 0
                    tpixels.append(pixels[i])
                else:
                    tpixels.append((repeat_count, 256, 256))
            else:
                repeat_count = 0
                tpixels.append(pixels[i])
        
        return tpixels
    
    def itransform_pixels(self, decoded_deltas, initial_pixel, size):
        deltas = []
        for i in range(1, len(decoded_deltas)):
            if decoded_deltas[i][1] == 256:
                for _ in range(decoded_deltas[i][0]):
                    deltas.append(decoded_deltas[i - 1])
            else:
                deltas.append(decoded_deltas[i])

        decoded_pixels = [initial_pixel] + deltas
        decoded_pixels = np.array(decoded_pixels).reshape(size[0], size[1], 3)
        decoded_pixels[:, 0, :] = np.cumsum(decoded_pixels[:, 0, :], axis=0)
        decoded_pixels = np.cumsum(decoded_pixels, axis=1)
        decoded_pixels = decoded_pixels.reshape(-1, 3)
        decoded_pixels = list(map(tuple, decoded_pixels))
        return decoded_pixels

    def key_to_int(self, rgb):
        """Map RGB tuple from [-255, 255] to a 27-bit integer."""
        r, g, b = [v + 255 for v in rgb]  # map -255..255 → 0..510
        return (r << 18) | (g << 9) | b

    def int_to_key(self, val):
        """Decode 27-bit int back to RGB tuple in range [-255, 255]."""
        r = ((val >> 18) & 0x1FF) - 255
        g = ((val >> 9) & 0x1FF) - 255
        b = (val & 0x1FF) - 255
        return (r, g, b)

    def encode(self):
        code = self._encode()
        encoded_data = to_bytes(code)
        
        codebook = self.compress_codebook()
        json_obj = {
            "size": self.image.size,
            "initial_pixel": self.key_to_int(self.pixels[0]),  # still stored as tuple
            "bit_length": len(code),
            "codebook": codebook,
            "bitstream": encoded_data
        }

        img_ext = self.image_path.split('.')[-1]
        self.json_path = self.image_path.replace(f'.{img_ext}', '.json')
        with open(self.json_path, 'w') as f:
            json.dump(json_obj, f, separators=(',', ':'))  # compact JSON
    
    def decode(self):
        """Decode the image data from a compressed JSON Huffman file."""
        with open(self.image_path, 'r') as f:
            data = json.load(f)

        self.data = data
        self.size = data["size"]
        self.codebook = data["codebook"]
        code_bytes = data["bitstream"]
        self.bits = from_bytes(code_bytes, data["bit_length"])

        self.initial_pixel = self.int_to_key(self.data["initial_pixel"])

        lookup_table = self.decompress_codebook(self.codebook)
        decoded_deltas = self._decode(self.bits, lookup_table)
        decoded_pixels = self.itransform_pixels(decoded_deltas, self.initial_pixel, self.size)

        img = Image.new("RGB", self.size)
        img.putdata(decoded_pixels)
        img.save(self.image_path.replace(".json", "_1.bmp"))
        return img


def run(image_path, method='encode'):
    if method == 'encode':
        encoder = HDelta(image_path)
        print(f"Alphabet size: {len(encoder.freq_map)}")

        entropy, expected_len = encoder.evaluate()
        print(f"Entropy (bits): {entropy:.4f}")
        print(f"Huffman expected length (bits): {expected_len:.4f}")

        size_bytes = os.path.getsize(image_path)
        size_bytes_enc = os.path.getsize(encoder.json_path)
        print(f"Original size: {size_bytes/1000:.2f} KB")
        print(f"Encoded size: {size_bytes_enc/1000:.2f} KB")
        print(f"Expected size of code: {size_bytes*expected_len/(24*1000):.2f} KB")

        print(f"Compression time: {encoder.compression_time:.4f} seconds")
        print(f"r/logT: {(size_bytes/size_bytes_enc)/encoder.compression_time}")

        img = Image.open(image_path)
        start_time = time.time()
        img.save(image_path.replace('.bmp', '.png'), format="PNG", optimize=True, compress_level=9)
        end_time = time.time()
        compression_time = end_time - start_time
        print(f"Compression time: {compression_time:.4f} seconds")
        size_bytes_png = os.path.getsize(image_path.replace('.bmp', '.png'))
        print(f"r/logT: {(size_bytes/size_bytes_png)/compression_time}")
    else:
        decoder = HDelta(image_path)


# Example usage
if __name__ == "__main__":
    img_path = "/home/naughtius-maximus/Desktop/ID/dp_linkedin.bmp"
    run(img_path, method='encode')
    run(img_path.replace('.bmp', '.json'), method='decode')
