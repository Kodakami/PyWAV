# pywav.py

# RIFF WAVE file format tools.

import struct
from enum import Enum

# Internal classes.

class SampleFormat:
    class Type(Enum):
        PCM_INTEGER = 1         # PCM unsigned integer samples (8-, 16-, 32-bit)
        PCM_FLOATING_POINT = 3  # PCM signed IEEE floating-point samples (32-, 64-bit)
    
    __FORMATS = {
        Type.PCM_INTEGER:{
        8:lambda float_sample : struct.pack("<B", SampleFormat.__float_to_uint(float_sample, 8)),
        16:lambda float_sample : struct.pack("<h", SampleFormat.__float_to_int(float_sample, 16)),
        32:lambda float_sample : struct.pack("<i", SampleFormat.__float_to_int(float_sample, 32))
        },
        Type.PCM_FLOATING_POINT:{
        32:lambda float_sample : struct.pack("<f", float_sample),
        64:lambda float_sample : struct.pack("<d", float_sample)
        }
    }

    __int_bit_depths = __FORMATS[Type.PCM_INTEGER].keys()
    __float_bit_depths = __FORMATS[Type.PCM_FLOATING_POINT].keys()
    
    def __init__(self, fmt_type:Type, bit_depth:int):
        if fmt_type == SampleFormat.Type.PCM_INTEGER and bit_depth not in SampleFormat.__int_bit_depths:
            raise Exception(f"Integer PCM samples must have one of the following bit depths: {[SampleFormat.__int_bit_depths]}.")
        elif fmt_type == SampleFormat.Type.PCM_FLOATING_POINT and bit_depth not in SampleFormat.__float_bit_depths:
            raise Exception(f"Floating-point PCM samples must have one of the following bit depths: {SampleFormat.__float_bit_depths}.")

        self.id = fmt_type.value
        self.bit_depth = bit_depth
        self.__format_method = SampleFormat.__FORMATS[fmt_type][bit_depth]

    def format(self, float_sample:float) -> bytes:
        return self.__format_method(float_sample)
            

    # class-level methods
    def __float_to_uint(float_sample:float, bit_depth:int) -> int:
        # Unsigned for 8-bit values. Nearest-neighbor rounding is handled manually.
        max_int_plus_one = 2**bit_depth
        max_int = max_int_plus_one - 1
        return int((((float_sample + 1) * max_int) / 2) + 0.5)

    def __float_to_int(float_sample:float, bit_depth:int) -> int:
        # Signed for 16-, and 32-bit values. Nearest-neighbor rounding is handled manually.
        max_int_plus_one = 2**bit_depth
        max_int = max_int_plus_one - 1
        half_int = int(max_int_plus_one / 2)
        return int((((float_sample + 1) * max_int) / 2) + 0.5) - half_int

    def int_fmt(bit_depth:int):
        return SampleFormat(SampleFormat.Type.PCM_INTEGER, bit_depth)
    def float_fmt(bit_depth:int):
        return SampleFormat(SampleFormat.Type.PCM_FLOATING_POINT, bit_depth)

# Constants.
RIFF = [0x52, 0x49, 0x46, 0x46]         # ['R', 'I', 'F', 'F'] in ASCII code points
HEADER_SIZE = 36                        # Size of header. ChunkSize is derived from this number + SubChunk2Size.
WAVE = [0x57, 0x41, 0x56, 0x45]         # ['W', 'A', 'V', 'E'] in ASCII code points
FMT_SPACE = [0x66, 0x6D, 0x74, 0x20]    # ['f', 'm', 't', ' '] in ASCII code points
DATA = [0x64, 0x61, 0x74, 0x61]         # ['d', 'a', 't', 'a'] in ASCII code points
SUBCHUNK1_SIZE = 16

# Functions
def __make_uint16(signed_int:int):
    return struct.pack("<H", signed_int)

def __make_uint32(signed_int:int):
    return struct.pack("<I", signed_int)

def __get_chunk_size(subchunk2_size:int):
    return __make_uint32(HEADER_SIZE + subchunk2_size)

def __get_num_channels(num_channels:int):
    return __make_uint16(num_channels)

def __get_sample_rate(sample_rate:int):
    return __make_uint32(sample_rate)

def __get_byte_rate(sample_rate:int, num_channels:int, bit_depth:int):
    return __make_uint32(int(sample_rate * num_channels * (bit_depth // 8)))

def __get_block_align(num_channels:int, bit_depth:int):
    return __make_uint16(int(num_channels * (bit_depth // 8)))

def __get_bit_depth(bit_depth:int):
    return __make_uint16(bit_depth)

# Also returns the size as a regular int for math purposes.
def __get_subchunk2_size(num_samples:int, bit_depth:int, num_channels:int):
    size = int(num_samples * num_channels * (bit_depth // 8))
    return size, __make_uint32(size)

def __create_from_samples(sample_rate:int, fmt:SampleFormat, num_channels:int, samples_by_channel:list):
    num_samples = -1
    
    # Testing arguments
    if sample_rate < 1 or sample_rate >= 2**32:
        raise Exception("Sample rate must be at least 1 sample per second.")
    if num_channels < 1 or num_channels >= 2**16:
        raise Exception("Number of channels must be positive and less than 65536.")
    if len(samples_by_channel) != num_channels:
        raise Exception("Samples must be provided for each channel.")
    else:
        for channel_samples in samples_by_channel:
            num_samples_in_channel = len(channel_samples)
            if num_samples == -1:
                num_samples = num_samples_in_channel
            elif num_samples_in_channel != num_samples:
                raise Exception("Each channel must have the same number of samples.")
    
    file_bytes = bytearray()
    
    # ChunkID
    file_bytes.extend(RIFF)

    # Need SubChunk2Size to continue.
    subchunk2_size, subchunk2_size_bytes = __get_subchunk2_size(num_samples, fmt.bit_depth, num_channels)

    # ChunkSize
    chunk_size_bytes = __get_chunk_size(subchunk2_size)
    file_bytes.extend(chunk_size_bytes)

    # Format
    file_bytes.extend(WAVE)

    # Subchunk1ID
    file_bytes.extend(FMT_SPACE)

    # Subchunk1Size
    subchunk1_size_bytes = __make_uint32(SUBCHUNK1_SIZE)
    file_bytes.extend(subchunk1_size_bytes)

    # AudioFormat
    audio_format_bytes = __make_uint16(fmt.id)
    file_bytes.extend(audio_format_bytes)

    # NumChannels
    num_channels_bytes = __get_num_channels(num_channels)
    file_bytes.extend(num_channels_bytes)

    # SampleRate
    sample_rate_bytes = __get_sample_rate(sample_rate)
    file_bytes.extend(sample_rate_bytes)

    # ByteRate
    byte_rate_bytes = __get_byte_rate(sample_rate, num_channels, fmt.bit_depth)
    file_bytes.extend(byte_rate_bytes)

    # BlockAlign
    block_align_bytes = __get_block_align(num_channels, fmt.bit_depth)
    file_bytes.extend(block_align_bytes)

    # BitsPerSample
    bits_per_sample_bytes = __get_bit_depth(fmt.bit_depth)
    file_bytes.extend(bits_per_sample_bytes)

    # SubChunk2ID
    file_bytes.extend(DATA)

    # SubChunk2Size (determined above)
    file_bytes.extend(subchunk2_size_bytes)

    # Data
    for sample_index in range(num_samples):
        for channel_index in range(num_channels):
            sample = samples_by_channel[channel_index][sample_index]
            if -1 <= sample <= 1:
                file_bytes.extend(fmt.format(sample))
            else:
                raise Exception(f"A sample value was out of range (value: {sample}). Samples must be between -1 and 1 (inclusive).")
    
    # Conditional padding byte if SubChunk2Size is odd.
    if subchunk2_size % 2 == 1:
        file_bytes.append(0x00)

    # End of File

    return file_bytes

def create_from_samples_mono(sample_rate:int, fmt:SampleFormat, samples:list) -> bytearray:
    return __create_from_samples(sample_rate, fmt, 1, [samples])

def create_from_samples_stereo(sample_rate:int, fmt:SampleFormat, left_samples:list, right_samples:list) -> bytearray:
    return __create_from_samples(sample_rate, fmt, 2, [left_samples, right_samples])