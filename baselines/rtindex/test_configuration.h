
#include "default_test_configuration.h"
#include "test_configuration_override.h"


#if NUM_UPDATES_LOG > 0 && LOCAL_UPDATE_CHUNK_SIZE_LOG > NUM_UPDATES_LOG
#error cannot have more local updates than the total number of updates
#endif

#if NUM_UPDATES_LOG > 0 && UPDATE_TYPE != 0 && LOCAL_UPDATE_CHUNK_SIZE_LOG == 0
#error cannot have a local update batch size of 1
#endif

#if LARGE_KEYS != 0
#define KEY_DECOMPOSITION LARGE_KEY_DECOMPOSITION
#else
#define KEY_DECOMPOSITION SMALL_KEY_DECOMPOSITION
#endif

#if NUM_BUILD_KEYS_LOG <= NUM_UPDATES_LOG
#error cannot have more updates than inserted keys
#endif

#if MULTIPLE_HITS_PER_RAY == 0 && RANGE_QUERY_HIT_COUNT > 0
#error code needs to handle multiple hits per ray when processing range queries
#endif

#if MULTIPLE_HITS_PER_RAY == 0 && KEY_REPLICATION_LOG > 0
#error code needs to handle multiple hits per ray when key replication is active
#endif

#if MULTIPLE_HITS_PER_RAY != 0 && USE_CLOSESTHIT_INSTEAD_OF_ANYHIT != 0
#error cannot handle multiple hits per ray when using closest-hit
#endif

#if RANGE_QUERY_HIT_COUNT > 0 && KEY_REPLICATION_LOG > 0
#error cannot generate range queries with replicated keys at the moment
#endif

#if RANGE_QUERY_HIT_COUNT > 0 && FORCE_UNIFORM_KEYS != 0
#error cannot generate range queries with uniform keys at the moment
#endif

#if USE_CLOSESTHIT_INSTEAD_OF_ANYHIT == 1 && RANGE_QUERY_HIT_COUNT > 0
#error using closest-hit instead of any-hit does not work in range query mode
#endif

#if MISS_PERCENTAGE != 0 && RANGE_QUERY_HIT_COUNT > 0
#error miss percentage does not work in range query mode
#endif

#if OUT_OF_RANGE_PERCENTAGE != 0 && RANGE_QUERY_HIT_COUNT > 0
#error out of range percentage does not work in range query mode
#endif

#if MISS_PERCENTAGE + OUT_OF_RANGE_PERCENTAGE > 100
#error cannot generate more than 100% combined misses
#endif

#if MISS_PERCENTAGE + OUT_OF_RANGE_PERCENTAGE > 0 && LEAVE_GAPS_FOR_MISSES == 0
#error cannot generate misses if there are no gaps
#endif

#if MISS_PERCENTAGE + OUT_OF_RANGE_PERCENTAGE == 0 && LEAVE_GAPS_FOR_MISSES != 0
#warning generating gaps without misses
#endif

#if PRIMITIVE == 1 && INT_TO_FLOAT_CONVERSION_MODE == 1
#error spheres do not work with exclusive key range
#endif

#if PRIMITIVE == 1 && INT_TO_FLOAT_CONVERSION_MODE == 2
#error spheres do not work with extended key range
#endif

#if PRIMITIVE == 2 && INT_TO_FLOAT_CONVERSION_MODE == 1
#error aabbs do not work with exclusive key range
#endif

#if START_RAY_AT_ZERO == 0 && INT_TO_FLOAT_CONVERSION_MODE == 2
#error ray has to be started at zero when using extended key range
#endif
