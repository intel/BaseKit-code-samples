#!/usr/bin/env python3
############################################################################
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT
############################################################################
"""Demonstration of simple video encode."""
import sys
import time
import argparse
import vpl

SUCCESS = 0
FAILURE = -1


def align_up(value, step):
    """Round a value up to the nearest multiple of step"""
    if value % step:
        value = ((value // step) * step) + step
    return value


def main():
    """Program entry point"""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="NV12 file")
    parser.add_argument("width", type=int)
    parser.add_argument("height", type=int)
    args = parser.parse_args()
    with open("out.h264", "wb") as output_stream:
        return encode_file(args.input, args.width, args.height, output_stream)
        output_stream.flush()


def encode_file(input, width, height, dest):
    """Main encode function"""
    status = FAILURE
    LogTrace("Creating H.264 encoder using default device (GPU if available)")
    encoder = vpl.create_workstream(vpl.TargetDevice.GPU_GEN)
    vpl.set_config_property(encoder, vpl.WorkstreamProp.WORKSTREAM_TYPE,
                            vpl.WorkstreamType.ENCODE)
    vpl.set_config_property(encoder, vpl.WorkstreamProp.DST_BITSTREAM_FORMAT,
                            vpl.FourCC.H264)

    vpl.set_config_property(encoder, vpl.WorkstreamProp.OUTPUT_RESOLUTION,
                            vpl.VideoSurfaceResolution(width, height))
    input_file = vpl.open_file(input, "rb")
    frame_count = 0
    timer = Timer()
    raw_buffer_size = (width * height * 3) / 2
    total_encoded_bytes = 0
    encode_done = False

    # Record format of incomming frames
    info = vpl.memory.ImageInfo()
    info.width = width
    info.height = height
    info.aligned_width = align_up(width, 16)
    info.aligned_height = align_up(height, 16)
    info.format = vpl.memory.PixelFormat.NV12

    LogTrace("Entering main encode loop")
    while not encode_done:
        encoded_bytes = None
        state = vpl.workstream_get_state(encoder)
        if state == vpl.WorkstreamState.READ_INPUT:
            raw_image = vpl.read_data(input_file, info)
            if raw_image:
                frame_count += 1
                print("Frame: {}".format(frame_count), file=sys.stderr)
            timer.start()
            vpl.memory.ref(raw_image);
            encoded_bytes = vpl.encode_frame(encoder, raw_image)
            timer.stop()
            if not raw_image:
                encode_done = len(encoded_bytes) == 0
                if encode_done:
                    status = SUCCESS
        elif state == vpl.WorkstreamState.END_OF_OPERATION:
            # The encoder has completed operation, and has no frames left to give.
            LogTrace("Encode complete")
            encode_done = True
            status = SUCCESS
        elif state == vpl.WorkstreamState.ERROR:
            LogTrace("Error during encode. Exiting.")
            encode_done = True
            status = FAILURE

        if encoded_bytes:
            total_encoded_bytes += len(encoded_bytes)
            dest.write(encoded_bytes)

    LogTrace("Frames encoded   : {}".format(frame_count))
    LogTrace("Frames per second: {:02f}".format(frame_count / timer.elapsed()))
    ratio = total_encoded_bytes / (frame_count * raw_buffer_size)
    LogTrace("Compression Ratio   : {}".format(ratio))
    return status


class Timer:
    """Simple timer that tracks total time elapsed between starts and stops"""
    def __init__(self):
        self._elapsed_time = 0
        self._start_time = 0
        self._stop_time = 0

    def start(self):
        """Start Timing"""
        self._start_time = time.time()

    def stop(self):
        """Stop Timing"""
        self._stop_time = time.time()
        self._elapsed_time += self._stop_time - self._start_time

    def elapsed(self):
        """Get Total time tracked"""
        return self._elapsed_time


def LogTrace(msg, *args, **kwargs):
    """Print message to stderr"""
    print(msg.format(*args, **kwargs), file=sys.stderr)


if __name__ == "__main__":
    main()
