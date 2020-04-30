#!/usr/bin/env python3
############################################################################
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT
############################################################################
"""Demonstration of simple video decode."""
import sys
import time
import os
import vpl
import argparse
import ctypes

# tkinter and Pillow are used for image display rendering,
try:
    import tkinter
    from PIL import Image, ImageTk
except ImportError:
    pass

SUCCESS = 0
FAILURE = -1

# Number of bytes to read from file at a time and pass to decoder
CHUNK_SIZE = 1024 * 1024


def main():
    """Program entry point"""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=argparse.FileType("rb"))
    args = parser.parse_args()
    status = SUCCESS
    window = VideoWindow()
    status = decode_and_render_file(args.input, window)
    window.close()
    return status


def decode_and_render_file(in_file, window):
    """Main decode and render function"""
    status = FAILURE
    LogTrace("Creating H.264 decoder using default device (GPU if available)")
    decoder = vpl.create_workstream(vpl.TargetDevice.DEFAULT)
    vpl.set_config_property(decoder, vpl.WorkstreamProp.WORKSTREAM_TYPE,
                            vpl.WorkstreamType.DECODEVIDEOPROC)
    vpl.set_config_property(decoder, vpl.WorkstreamProp.SRC_BITSTREAM_FORMAT,
                            vpl.FourCC.H264)

    LogTrace("Setting target format and color-space (CSC).")
    vpl.set_config_property(decoder, vpl.WorkstreamProp.DST_RAW_FORMAT,
                            vpl.FourCC.BGRA)

    LogTrace("Setting target resolution (scaling).")
    vpl.set_config_property(decoder, vpl.WorkstreamProp.OUTPUT_RESOLUTION,
                            vpl.VideoSurfaceResolution(352, 288))

    frame_count = 0
    timer = Timer()
    decode_done = False

    LogTrace("Entering main decode loop")
    while not decode_done:
        image = None
        state = vpl.workstream_get_state(decoder)
        if state == vpl.WorkstreamState.READ_INPUT:
            # The decoder can accept more data, read it from file and pass it in.
            data = in_file.read(CHUNK_SIZE)
            timer.start()
            image = vpl.decode_process_frame(decoder, data)
            timer.stop()

        elif state == vpl.WorkstreamState.INPUT_BUFFER_FULL:
            # The decoder cannot accept more data, call DecodeFrame to drain.
            timer.start()
            image = vpl.decode_process_frame(decoder, None)
            timer.stop()

        elif state == vpl.WorkstreamState.END_OF_OPERATION:
            # The decoder has completed operation, and has no frames left to give.
            LogTrace("Decode complete")
            decode_done = True
            status = SUCCESS

        elif state == vpl.WorkstreamState.ERROR:
            LogTrace("Error during decode. Exiting.")
            decode_done = True
            status = FAILURE

        if image:
            # DecodeFrame returned a frame, use it.
            frame_count += 1
            print("Frame: {}".format(frame_count), file=sys.stderr)
            window.show(image)
            # Release the reference to the frame, so the memory can be reclaimed
            vpl.memory.unref(image)

    LogTrace("Frames decoded   : {}".format(frame_count))
    LogTrace("Frames per second: {:02f}".format(frame_count / timer.elapsed()))
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


def disable_event():
    pass


class VideoWindow():
    """Window to render frames"""
    def __init__(self):
        self.root = None
        self.canvas = None
        self.canvas_image = None
        self.use_gui = True
        if os.name != "nt":
            if "DISPLAY" not in os.environ or not os.environ["DISPLAY"]:
                self.use_gui = False
        if "tkinter" not in sys.modules:
            LogTrace("tkinter not found")
            self.use_gui = False
        if "PIL" not in sys.modules:
            LogTrace("Pillow not found")
            self.use_gui = False
        if not self.use_gui:
            LogTrace("Display unavailable, continuing without...")
            return
        self.root = tkinter.Tk()
        self.root.protocol("WM_DELETE_WINDOW", disable_event)
        self.root.title("Display decoded output")
        self.canvas = None
        self.canvas_image = None


    def show(self, image):
        """Render frame to window"""
        if not self.use_gui:
            return
        desc = vpl.memory.get_image_info(image)
        width = desc.width
        height = desc.height
        stride = width * 4
        image_data = vpl.memory.map_image(image, vpl.memory.AccessFlags.READ)
        c_image_plane_data = ctypes.POINTER(ctypes.c_uint8 *
                                            (stride * image_data.info.height))
        data = ctypes.cast(image_data.planes[0].data,
                           c_image_plane_data).contents
        img = Image.frombytes("RGB", (width, height), data, "raw", "BGRX",
                              stride)
        photo_img = ImageTk.PhotoImage(image=img)
        vpl.memory.unmap_image(image_data)
        if self.canvas is None:
            self.canvas = tkinter.Canvas(self.root, width=width, height=height)
            self.canvas.pack()
            self.canvas_image = self.canvas.create_image((0, 0),
                                                         image=photo_img,
                                                         anchor="nw")
        else:
            self.canvas.itemconfig(self.canvas_image, image=photo_img)
        self.root.update()

    def close(self):
        """Close window"""
        if self.root:
            self.root.destroy()


def LogTrace(msg, *args, **kwargs):
    """Print message to stderr"""
    print(msg.format(*args, **kwargs), file=sys.stderr)


if __name__ == "__main__":
    main()
