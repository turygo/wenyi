#include "bridge.h"
#include <stdlib.h>
#include <limits.h>
#include <unicode/ucnv.h>
#include <unicode/ucnv_err.h>
#include <unicode/udata.h>
#include <unicode/uversion.h>

int wenyi_encoding_init(void) {
    UErrorCode error = U_ZERO_ERROR;
    udata_setFileAccess(UDATA_NO_FILES, &error);
    return U_SUCCESS(error) ? 0 : 3;
}

int wenyi_convert(const char *bytes, int length, const char *from, const char *to,
                  int replace, char **output, int *output_length) {
    *output = NULL; *output_length = 0;
    if (length < 0) return 2;
    UErrorCode error = U_ZERO_ERROR;
    UConverter *source = ucnv_open(from, &error);
    if (U_FAILURE(error) || !source) return 3;
    UConverter *target = ucnv_open(to, &error);
    if (U_FAILURE(error) || !target) { ucnv_close(source); return 3; }
    ucnv_setToUCallBack(source, replace ? UCNV_TO_U_CALLBACK_SUBSTITUTE : UCNV_TO_U_CALLBACK_STOP, NULL, NULL, NULL, &error);
    ucnv_setFromUCallBack(target, replace ? UCNV_FROM_U_CALLBACK_SUBSTITUTE : UCNV_FROM_U_CALLBACK_STOP, NULL, NULL, NULL, &error);
    int32_t required = ucnv_toUChars(source, NULL, 0, bytes, length, &error);
    if (error != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(error)) { ucnv_close(source); ucnv_close(target); return 3; }
    if (required < 0 || required == INT32_MAX || (size_t)required + 1 > SIZE_MAX / sizeof(UChar)) { ucnv_close(source); ucnv_close(target); return 2; }
    error = U_ZERO_ERROR;
    UChar *unicode = (UChar *)malloc(((size_t)required + 1) * sizeof(UChar));
    if (!unicode) { ucnv_close(source); ucnv_close(target); return 2; }
    ucnv_reset(source);
    int32_t actual = ucnv_toUChars(source, unicode, required + 1, bytes, length, &error);
    if (U_FAILURE(error)) { free(unicode); ucnv_close(source); ucnv_close(target); return 3; }
    error = U_ZERO_ERROR;
    int32_t capacity = ucnv_fromUChars(target, NULL, 0, unicode, actual, &error);
    if ((error != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(error)) || capacity < 0 || capacity == INT32_MAX) {
        free(unicode); ucnv_close(source); ucnv_close(target); return U_FAILURE(error) && error != U_BUFFER_OVERFLOW_ERROR ? 3 : 2;
    }
    error = U_ZERO_ERROR;
    char *result = (char *)malloc((size_t)capacity + 1);
    if (!result) { free(unicode); ucnv_close(source); ucnv_close(target); return 2; }
    ucnv_reset(target);
    int32_t produced = ucnv_fromUChars(target, result, capacity + 1, unicode, actual, &error);
    free(unicode); ucnv_close(source); ucnv_close(target);
    if (U_FAILURE(error)) { free(result); return 3; }
    *output = result; *output_length = produced;
    return 0;
}
void wenyi_free(void *pointer) { free(pointer); }
void wenyi_icu_version(char output[32]) {
    _Static_assert(U_MAX_VERSION_STRING_LENGTH <= 32, "ICU version buffer ABI");
    UVersionInfo version;
    u_getVersion(version);
    u_versionToString(version, output);
}
