// ==============================================================
// Copyright (C) 2019 Intel Corporation
// 
// SPDX-License-Identifier: MIT
// =============================================================
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

#ifndef __KERNELS_H__
#define __KERNELS_H__
#pragma once

// VECPOW == 2 means VEC == 4.
// VECPOW == 3 means VEC == 8.
// VECPOW == 4 means VEC == 16.
#define VECPOW 4 

#define VEC (1<<VECPOW) 
#define VECX2 (2 * VEC)

#define HUFTABLESIZE 256

//Maximum length of huffman codes
#define MAX_HUFFCODE_BITS 16

struct uint2 {
    unsigned int y;
    unsigned int x;
};

struct lz_input_t {
  unsigned char data[VEC];
};

typedef struct _dist_len_t {
    unsigned char data[VEC]; 
    char         len[VEC];
    short         dist[VEC];
} dist_len_t, *pdist_len_t;

struct huffman_output_t {
  unsigned int data[VEC];
  bool write;
};

struct _trailing_output_t {
    int bytecount_left;
    int bytecount; 
    unsigned char bytes[VEC*sizeof(unsigned int)];
};

struct gzip_out_info_t {
  // final compressed block size
  size_t compression_sz;
  unsigned long crc;
};

// LEN must be == VEC
#define LEN (VEC)

// depth of the dictionary buffers
#define DEPTH 512

// Assumes DEPTH is a power of 2 number.
#define HASH_MASK (DEPTH-1)

#define CONSTANT __constant

#define DEBUG 1
#define TRACE(x) do { if (DEBUG) printf x; } while (0)

#define STATIC_TREES 1

typedef struct ct_data {
    unsigned short code;
    unsigned short len;
} ct_data;

#define MAX_MATCH 258
#define MIN_MATCH  3

#define TOO_FAR 4096

// All codes must not exceed MAX_BITS
#define MAX_BITS 15

// number of length codes, not counting the special END_BLOCK code
#define LENGTH_CODES 29

// number of literal bytes, 0..255
#define LITERALS 256

// end of literal code block
#define END_BLOCK 256

// number of literal or length codes, including END_BLOCK
#define L_CODES (LITERALS+1+LENGTH_CODES)

// number of distance codes
#define D_CODES 30

// number of codes used to transfer the bit lengths
#define BL_CODES 19

#define MAX_DISTANCE ((32*1024)) 

struct dict_string {
  unsigned char s[LEN];
};

// Mapping from a distance to a distance code. dist is the distance - 1 and
// must not have side effects. dist_code[256] and dist_code[257] are never
// used.
#define d_code(dist) \
   ((dist) < 256 ? dist_code[dist] : dist_code[256+((dist)>>7)])

#endif //__KERNELS_H__
