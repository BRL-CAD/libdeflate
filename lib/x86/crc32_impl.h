/*
 * x86/crc32_impl.h - x86 implementations of the gzip CRC-32 algorithm
 *
 * Copyright 2016 Eric Biggers
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef LIB_X86_CRC32_IMPL_H
#define LIB_X86_CRC32_IMPL_H

#include "cpu_features.h"

/* PCLMUL implementation */
#if HAVE_PCLMUL_INTRIN
#  define SUFFIX			 _pclmul
#  define crc32_x86_pclmul	crc32_x86_pclmul
#  define FOLD_PARTIAL_VECS	0
#  if HAVE_PCLMUL_NATIVE
#    define ATTRIBUTES
#  else
#    define ATTRIBUTES		__attribute__((target("pclmul")))
#  endif
#  include "crc32_pclmul_template.h"
#endif

/*
 * PCLMUL/AVX implementation.  This implementation has two benefits over the
 * regular PCLMUL one.  First, simply compiling against the AVX target can
 * improve performance significantly (e.g. 10100 MB/s to 16700 MB/s on Skylake)
 * without any code changes, probably due to the availability of non-destructive
 * VEX-encoded instructions.  Second, AVX support implies SSSE3 and SSE4.1
 * support, and we can use SSSE3 and SSE4.1 intrinsics for efficient handling of
 * partial blocks.  (For simplicity, we don't currently bother compiling a
 * variant with PCLMUL+SSSE3+SSE4.1 without AVX.)
 */
#if HAVE_PCLMUL_INTRIN && HAVE_AVX_INTRIN && \
	((HAVE_PCLMUL_NATIVE && HAVE_AVX_NATIVE) || \
	 (HAVE_PCLMUL_TARGET && HAVE_AVX_TARGET))
#  define SUFFIX			 _pclmul_avx
#  define crc32_x86_pclmul_avx	crc32_x86_pclmul_avx
#  define FOLD_PARTIAL_VECS	1
#  if HAVE_PCLMUL_NATIVE && HAVE_AVX_NATIVE
#    define ATTRIBUTES
#  else
#    define ATTRIBUTES		__attribute__((target("pclmul,avx")))
#  endif
#  include "crc32_pclmul_template.h"
#endif

#if HAVE_VPCLMULQDQ_INTRIN && HAVE_PCLMUL_INTRIN && HAVE_AVX2_INTRIN
#  if HAVE_VPCLMULQDQ_NATIVE && HAVE_PCLMUL_NATIVE && HAVE_AVX2_NATIVE
#    define ATTRIBUTES
#  else
#    define ATTRIBUTES		__attribute__((target("vpclmulqdq,pclmul,avx2,avx512vl")))
#  endif
static forceinline ATTRIBUTES __m128i
fold_vec128(__m128i src, __m128i dst, __v2di multipliers)
{
	return dst ^ _mm_clmulepi64_si128(src, multipliers, 0x00) ^
		_mm_clmulepi64_si128(src, multipliers, 0x11);
}

static forceinline ATTRIBUTES __m256i
fold_vec256(__m256i src, __m256i dst, __v4di multipliers)
{
	__m256i a = _mm256_clmulepi64_epi128(src, multipliers, 0x00);
	__m256i b = _mm256_clmulepi64_epi128(src, multipliers, 0x11);

#if 0
	return _mm256_ternarylogic_epi64(a, b, dst, 0x96);
#else
	return dst ^ a ^ b;
#endif
}

#define crc32_x86_pclmul256_avx2	crc32_x86_pclmul256_avx2
static u32 ATTRIBUTES MAYBE_UNUSED
crc32_x86_pclmul256_avx2(u32 crc, const u8 *p, size_t len)
{
	const __v4di multipliers_16 = (__v4di)
		{ CRC32_16VECS_MULT_1, CRC32_16VECS_MULT_2,
		  CRC32_16VECS_MULT_1, CRC32_16VECS_MULT_2 };
	const __v4di multipliers_8 = (__v4di)
		{ CRC32_8VECS_MULT_1, CRC32_8VECS_MULT_2,
		  CRC32_8VECS_MULT_1, CRC32_8VECS_MULT_2 };
	const __v4di multipliers_4 = (__v4di)
		{ CRC32_4VECS_MULT_1, CRC32_4VECS_MULT_2,
		  CRC32_4VECS_MULT_1, CRC32_4VECS_MULT_2 };
	const __v4di multipliers_2 = (__v4di)
		{ CRC32_2VECS_MULT_1, CRC32_2VECS_MULT_2,
		  CRC32_2VECS_MULT_1, CRC32_2VECS_MULT_2 };
	const __v2di multipliers_1 = (__v2di)CRC32_1VECS_MULTS;
	const __v2di final_multiplier = (__v2di){ CRC32_FINAL_MULT };
	const __m128i mask32 = (__m128i)(__v4si){ 0xFFFFFFFF };
	const __v2di barrett_reduction_constants = (__v2di)CRC32_BARRETT_CONSTANTS;
	__m256i y0, y1, y2, y3, y4, y5, y6, y7;
	__m128i x0, x1;

	/*
	 * There are two overall code paths.  The first path supports all
	 * lengths, but is intended for short lengths; it uses unaligned loads
	 * and does at most 4-way folds.  The second path only supports longer
	 * lengths, aligns the pointer in order to do aligned loads, and does up
	 * to 8-way folds.  The length check below decides which path to take.
	 */
	if (len < 1024) {
		return crc32_slice1(crc, p, len);

		y0 = _mm256_loadu_si256((const void *)p) ^ (__m256i)(__v8si){crc};
		p += 32;
	} else {
		const size_t align = -(uintptr_t)p & 31;
		const __m256i *yp;

		if (align) {
			crc = crc32_slice1(crc, p, align);
			p += align;
			len -= align;
		}
		yp = (const __m256i *)p;
		y0 = *yp++ ^ (__m256i)(__v8si){crc};
		y1 = *yp++;
		y2 = *yp++;
		y3 = *yp++;
		y4 = *yp++;
		y5 = *yp++;
		y6 = *yp++;
		y7 = *yp++;
		do {
			y0 = fold_vec256(y0, *yp++, multipliers_16);
			y1 = fold_vec256(y1, *yp++, multipliers_16);
			y2 = fold_vec256(y2, *yp++, multipliers_16);
			y3 = fold_vec256(y3, *yp++, multipliers_16);
			y4 = fold_vec256(y4, *yp++, multipliers_16);
			y5 = fold_vec256(y5, *yp++, multipliers_16);
			y6 = fold_vec256(y6, *yp++, multipliers_16);
			y7 = fold_vec256(y7, *yp++, multipliers_16);
			len -= 256;
		} while (len >= 256 + 256);

		y0 = fold_vec256(y0, y4, multipliers_8);
		y1 = fold_vec256(y1, y5, multipliers_8);
		y2 = fold_vec256(y2, y6, multipliers_8);
		y3 = fold_vec256(y3, y7, multipliers_8);
		if (len & 128) {
			y0 = fold_vec256(y0, *yp++, multipliers_8);
			y1 = fold_vec256(y1, *yp++, multipliers_8);
			y2 = fold_vec256(y2, *yp++, multipliers_8);
			y3 = fold_vec256(y3, *yp++, multipliers_8);
		}

		y0 = fold_vec256(y0, y2, multipliers_4);
		y1 = fold_vec256(y1, y3, multipliers_4);
		if (len & 64) {
			y0 = fold_vec256(y0, *yp++, multipliers_4);
			y1 = fold_vec256(y1, *yp++, multipliers_4);
		}
		y0 = fold_vec256(y0, y1, multipliers_2);
		if (len & 32)
			y0 = fold_vec256(y0, *yp++, multipliers_2);

		x0 = fold_vec128(_mm256_extractf128_si256(y0, 0),
				 _mm256_extractf128_si256(y0, 1),
				 multipliers_1);
		p = (const u8 *)yp;
		if (len & 16) {
			x0 = fold_vec128(x0, *(__m128i *)p, multipliers_1);
			p += 16;
		}
	}
	len &= 15;

	/* Fold 128 => 96 bits, also implicitly appending 32 zero bits. */
	x0 = _mm_srli_si128(x0, 8) ^
	     _mm_clmulepi64_si128(x0, multipliers_1, 0x10);

	/* Fold 96 => 64 bits. */
	x0 = _mm_srli_si128(x0, 4) ^
	     _mm_clmulepi64_si128(x0 & mask32, final_multiplier, 0x00);

	/* Reduce 64 => 32 bits using Barrett reduction. */
	x1 = _mm_clmulepi64_si128(x0 & mask32, barrett_reduction_constants, 0x00);
	x1 = _mm_clmulepi64_si128(x1 & mask32, barrett_reduction_constants, 0x10);
	crc = _mm_cvtsi128_si32(_mm_srli_si128(x0 ^ x1, 4));
	/* Process up to 15 bytes left over at the end. */
	crc = crc32_slice1(crc, p, len);
	return crc;
}
#endif /* crc32_x86_pclmul256_avx2() */

/*
 * If the best implementation is statically available, use it unconditionally.
 * Otherwise choose the best implementation at runtime.
 */
#if defined(crc32_x86_pclmul256_avx2) && \
	HAVE_VPCLMULQDQ_NATIVE && HAVE_PCLMUL_NATIVE && HAVE_AVX2_NATIVE
#define DEFAULT_IMPL	crc32_x86_pclmul256_avx2
#else
static inline crc32_func_t
arch_select_crc32_func(void)
{
	const u32 features MAYBE_UNUSED = get_x86_cpu_features();

#ifdef crc32_x86_pclmul256_avx2
	if (HAVE_VPCLMULQDQ(features))
		return crc32_x86_pclmul256_avx2;
#endif
#ifdef crc32_x86_pclmul_avx
	if (HAVE_PCLMUL(features) && HAVE_AVX(features))
		return crc32_x86_pclmul_avx;
#endif
#ifdef crc32_x86_pclmul
	if (HAVE_PCLMUL(features))
		return crc32_x86_pclmul;
#endif
	return NULL;
}
#define arch_select_crc32_func	arch_select_crc32_func
#endif

#endif /* LIB_X86_CRC32_IMPL_H */
