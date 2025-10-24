
// rays should be cast perpendicularly to the line of triangles (point query only)
#define PERPENDICULAR_RAYS 1
// enable BHV compaction
#define COMPACTION 1
// set the OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL flag for BHV
#define FORCE_SINGLE_ANYHIT 1
// use the closest-hit shader instead of the any-hit shader to process point query hits
#define USE_CLOSESTHIT_INSTEAD_OF_ANYHIT 0
// start rays at zero and limit the hit range using tmin/tmax (range query only)
#define START_RAY_AT_ZERO 0
// enable 64bit keys
#define LARGE_KEYS 0
// reserve some keys of the build set, these keys can be probed later to simulate misses
#define LEAVE_GAPS_FOR_MISSES 0
// device code should handle multiple hits by aggregating the values, e.g. for range queries or key replication
#define MULTIPLE_HITS_PER_RAY 0
// skip the probing step altogether
#define SKIP_PROBING 0
// draw keys uniformly from the entire range
#define FORCE_UNIFORM_KEYS 0

// VALUES: 0 (global), 1 (position-local), 2 (rank-local)
#define UPDATE_TYPE 0

// VALUES: 0 (triangle), 1 (sphere), 2 (aabb)
#define PRIMITIVE 0

// VALUES: -1 (descending), 0 (shuffle), 1 (ascending)
#define INSERT_SORTED 0
// VALUES: -1 (descending), 0 (shuffle), 1 (ascending), 2 (sort on gpu)
#define PROBE_SORTED 0

// a bias to add to the exponent during key conversion
// this is done to check whether scaling all keys impacts performance
#define EXPONENT_BIAS 0

// in 3d mode, this is the key decomposition configuration (6-digit number)
// first two digits: number of least significant bits to use for the x coordinate
// second two digits: number of bits to use for the y coordinate
// last two digits: number of most significant bits to use for the z coordinate
// each number has to be < 24 and all numbers have to add up to the key size (32 or 64)
// example: 230900 means "use 23 bits for x, 9 for y, and z is always zero"
// example: 232318 means "use 23 bits for x, 23 for y, and 18 for z"
#define LARGE_KEY_DECOMPOSITION 232318
#define SMALL_KEY_DECOMPOSITION 230900

// how many updates should be performed on the BVH (0 = disable)
#define NUM_UPDATES_LOG 0
// size of the sub-array to shuffle locally
#define LOCAL_UPDATE_CHUNK_SIZE_LOG 1

#define NUM_BUILD_KEYS_LOG 26
#define NUM_PROBE_KEYS_LOG 27
#define NUM_RAYS_PER_THREAD_LOG 0
#define KEY_REPLICATION_LOG 0
#define RANGE_QUERY_HIT_COUNT_LOG 0
#define MISS_PERCENTAGE 0
#define OUT_OF_RANGE_PERCENTAGE 0
#define KEY_STRIDE_LOG 0

// VALUES: 0 (safe), 1 (exclusive), 2 (extended), 3 (3d)
#define INT_TO_FLOAT_CONVERSION_MODE 3
