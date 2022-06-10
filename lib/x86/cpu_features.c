/*
 * x86/cpu_features.c - feature detection for x86 CPUs
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

#include "../cpu_features_common.h" /* must be included first */
#include "cpu_features.h"

#if HAVE_DYNAMIC_X86_CPU_FEATURES

/* With old GCC versions we have to manually save and restore the x86_32 PIC
 * register (ebx).  See: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=47602  */
#if defined(__i386__) && defined(__PIC__)
#  define EBX_CONSTRAINT "=&r"
#else
#  define EBX_CONSTRAINT "=b"
#endif

/* Execute the CPUID instruction.  */
static inline void
cpuid(u32 leaf, u32 subleaf, u32 *a, u32 *b, u32 *c, u32 *d)
{
	__asm__(".ifnc %%ebx, %1; mov  %%ebx, %1; .endif\n"
		"cpuid                                  \n"
		".ifnc %%ebx, %1; xchg %%ebx, %1; .endif\n"
		: "=a" (*a), EBX_CONSTRAINT (*b), "=c" (*c), "=d" (*d)
		: "a" (leaf), "c" (subleaf));
}

/* Read an extended control register.  */
static inline u64
read_xcr(u32 index)
{
	u32 edx, eax;

	/* Execute the "xgetbv" instruction.  Old versions of binutils do not
	 * recognize this instruction, so list the raw bytes instead.  */
	__asm__ (".byte 0x0f, 0x01, 0xd0" : "=d" (edx), "=a" (eax) : "c" (index));

	return ((u64)edx << 32) | eax;
}

#undef BIT
#define BIT(nr)			(1UL << (nr))

#define XCR0_BIT_SSE		BIT(1)
#define XCR0_BIT_AVX		BIT(2)
#define XCR0_BIT_OPMASK		BIT(5)
#define XCR0_BIT_ZMM_HI256	BIT(6)
#define XCR0_BIT_HI16_ZMM	BIT(7)

#define IS_SET(reg, nr)		((reg) & BIT(nr))
#define IS_ALL_SET(reg, mask)	(((reg) & (mask)) == (mask))

static const struct cpu_feature x86_cpu_feature_table[] = {
	{X86_CPU_FEATURE_SSE2,		"sse2"},
	{X86_CPU_FEATURE_PCLMUL,	"pclmul"},
	{X86_CPU_FEATURE_AVX,		"avx"},
	{X86_CPU_FEATURE_AVX2,		"avx2"},
	{X86_CPU_FEATURE_BMI2,		"bmi2"},
	{X86_CPU_FEATURE_AVX512BW,	"avx512bw"},
};

volatile u32 libdeflate_x86_cpu_features = 0;

/*
 * Don't use the AVX-512 zmm registers without a runtime CPU model check, due to
 * the downclocking penalty on some CPUs.
 */
static bool zmm_allowlisted(char manufacturer[12], u32 family, u32 model)
{
#ifdef TEST_SUPPORT__DO_NOT_USE
	return true;
#else
	if (memcmp(manufacturer, "GenuineIntel", 12) == 0 && family == 6) {
		switch (model) {
		case 106: /* Ice Lake (Server) */
		case 125: /* Ice Lake (Client) */
		case 126: /* Ice Lake (Client) */
		case 167: /* Rocket Lake */
			return true;
		}
	}
	return false;
#endif
}

/* Initialize libdeflate_x86_cpu_features. */
void libdeflate_init_x86_cpu_features(void)
{
	u32 features = 0;
	u32 max_function;
	struct {
		u32 b;
		u32 d;
		u32 c;
	} manufacturer;
	u32 family_and_model;
	u32 dummy1, dummy2, dummy4;
	u32 features_1, features_2, features_3, features_4;
	u32 family;
	u32 model;
	bool ymm_allowed = false;
	bool zmm_allowed = false;

	/* Get maximum supported function  */
	cpuid(0, 0, &max_function, &manufacturer.b, &manufacturer.c,
	      &manufacturer.d);
	if (max_function < 1)
		goto out;

	/* Family, model, and standard feature flags */
	cpuid(1, 0, &family_and_model, &dummy2, &features_2, &features_1);

	family = (family_and_model >> 8) & 0xf;
	model = (family_and_model >> 4) & 0xf;
	if (family == 6 || family == 15) {
		model += (family_and_model >> 12) & 0xf0;
		if (family == 15)
			family += (family_and_model >> 20) & 0xff;
	}

	if (IS_SET(features_1, 26))
		features |= X86_CPU_FEATURE_SSE2;

	if (IS_SET(features_2, 1))
		features |= X86_CPU_FEATURE_PCLMUL;

	if (IS_SET(features_2, 27)) { /* OSXSAVE set? */
		u64 xcr0 = read_xcr(0);

		ymm_allowed = IS_ALL_SET(xcr0, XCR0_BIT_SSE | XCR0_BIT_AVX);

		zmm_allowed = IS_ALL_SET(xcr0, XCR0_BIT_SSE | XCR0_BIT_AVX |
					 XCR0_BIT_OPMASK | XCR0_BIT_ZMM_HI256 |
					 XCR0_BIT_HI16_ZMM) &&
			      zmm_allowlisted((char *)&manufacturer,
					      family, model);
	}

	if (ymm_allowed && IS_SET(features_2, 28))
		features |= X86_CPU_FEATURE_AVX;

	if (max_function < 7)
		goto out;

	/* Extended feature flags  */
	cpuid(7, 0, &dummy1, &features_3, &features_4, &dummy4);

	if (ymm_allowed && IS_SET(features_3, 5))
		features |= X86_CPU_FEATURE_AVX2;

	if (IS_SET(features_3, 8))
		features |= X86_CPU_FEATURE_BMI2;

	if (zmm_allowed && IS_SET(features_3, 30))
		features |= X86_CPU_FEATURE_AVX512BW;

out:
	disable_cpu_features_for_testing(&features, x86_cpu_feature_table,
					 ARRAY_LEN(x86_cpu_feature_table));

	libdeflate_x86_cpu_features = features | X86_CPU_FEATURES_KNOWN;
}

#endif /* HAVE_DYNAMIC_X86_CPU_FEATURES */
