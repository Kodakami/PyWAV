# pywav.py

# RIFF WAVE file format tools.

import sys

# Constants.
RIFF = [0x52, 0x49, 0x46, 0x46]         # ['R', 'I', 'F', 'F'] in ASCII code points
HEADER_SIZE = 36                        # Size of header. ChunkSize is derived from this number + SubChunk2Size.
WAVE = [0x57, 0x41, 0x56, 0x45]         # ['W', 'A', 'V', 'E'] in ASCII code points
FMT_SPACE = [0x66, 0x6D, 0x74, 0x20]    # ['f', 'm', 't', ' '] in ASCII code points
DATA = [0x64, 0x61, 0x74, 0x61]    # ['d', 'a', 't', 'a'] in ASCII code points
SUBCHUNK1_SIZE = 16
AUDIO_FORMAT_ID = 1

def __make_unsigned_int(signed_int:int, bit_count:int):
    # Grab each byte value little-endian.
    return bytes([(signed_int >> i) % 256 for i in range(0, bit_count, 8)])

def __get_chunk_size(subchunk2_size:int):
    return __make_unsigned_int(HEADER_SIZE + subchunk2_size, 32)

def __get_num_channels(num_channels:int):
    return __make_unsigned_int(num_channels, 16)

def __get_sample_rate(sample_rate:int):
    return __make_unsigned_int(sample_rate, 32)

def __get_byte_rate(sample_rate:int, num_channels:int, bits_per_sample:int):
    return __make_unsigned_int(int(sample_rate * num_channels * (bits_per_sample // 8)), 32)

def __get_block_align(num_channels:int, bits_per_sample:int):
    return __make_unsigned_int(int(num_channels * (bits_per_sample / 8)), 16)

def __get_bits_per_sample(bits_per_sample:int):
    return __make_unsigned_int(bits_per_sample, 16)

# Also returns the size as a regular int for math purposes.
def __get_subchunk2_size(num_samples:int, bits_per_sample:int, num_channels:int):
    size = int(num_samples * num_channels * (bits_per_sample / 8))
    return size, __make_unsigned_int(size, 32)

def __float_sample_to_int(float_sample:float, bits_per_sample:int):
    if -1 <= float_sample <= 1:
        return int(round(((float_sample / 2) + 0.5) * ((2**bits_per_sample) - 1)))
    raise Exception("Sample value out of range.")

def __create_from_samples(sample_rate:int, bits_per_sample:int, num_channels:int, samples_by_channel:list):
    num_samples = -1
    
    # Testing arguments
    if sample_rate < 1 or sample_rate >= 2**32:
        raise Exception("Sample rate must be at least 1 sample per second.")
    if bits_per_sample < 8 or bits_per_sample >= 2**16 or bits_per_sample % 8 != 0:
        raise Exception("Bits per sample must be a positive multiple of 8 less than 65536.")
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
    subchunk2_size, subchunk2_size_bytes = __get_subchunk2_size(num_samples, num_channels, bits_per_sample)

    # ChunkSize
    chunk_size_bytes = __get_chunk_size(subchunk2_size)
    file_bytes.extend(chunk_size_bytes)

    # Format
    file_bytes.extend(WAVE)

    # Subchunk1ID
    file_bytes.extend(FMT_SPACE)

    # Subchunk1Size
    subchunk1_size_bytes = __make_unsigned_int(SUBCHUNK1_SIZE, 32)
    file_bytes.extend(subchunk1_size_bytes)

    # AudioFormat
    audio_format_bytes = __make_unsigned_int(AUDIO_FORMAT_ID, 16)
    file_bytes.extend(audio_format_bytes)

    # NumChannels
    num_channels_bytes = __get_num_channels(num_channels)
    file_bytes.extend(num_channels_bytes)

    # SampleRate
    sample_rate_bytes = __get_sample_rate(sample_rate)
    file_bytes.extend(sample_rate_bytes)

    # ByteRate
    byte_rate_bytes = __get_byte_rate(sample_rate, num_channels, bits_per_sample)
    file_bytes.extend(byte_rate_bytes)

    # BlockAlign
    block_align_bytes = __get_block_align(num_channels, bits_per_sample)
    file_bytes.extend(block_align_bytes)

    # BitsPerSample
    bits_per_sample_bytes = __get_bits_per_sample(bits_per_sample)
    file_bytes.extend(bits_per_sample_bytes)

    # SubChunk2ID
    file_bytes.extend(DATA)

    # SubChunk2Size (determined above)
    file_bytes.extend(subchunk2_size_bytes)

    # Data
    for sample_index in range(num_samples):
        for channel_index in range(num_channels):
            file_bytes.extend(__make_unsigned_int(__float_sample_to_int(samples_by_channel[channel_index][sample_index], bits_per_sample), bits_per_sample))
    
    # End of File

    return file_bytes

def create_from_samples_mono(sample_rate:int, bits_per_sample:int, samples:list) -> bytearray:
    return __create_from_samples(sample_rate, bits_per_sample, 1, [samples])

def create_from_samples_stereo(sample_rate:int, bits_per_sample:int, left_samples:list, right_samples:list) -> bytearray:
    return __create_from_samples(sample_rate, bits_per_sample, 2, [left_samples, right_samples])