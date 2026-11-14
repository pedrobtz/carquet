/**
 * @file geo_wkb.c
 * @brief WKB walker for Parquet GeospatialStatistics (see geo_wkb.h).
 */

#include "core/geo_wkb.h"
#include <math.h>
#include <string.h>

void carquet_geo_stats_init(parquet_geospatial_statistics_t* s) {
    memset(s, 0, sizeof(*s));
}

static void add_type(parquet_geospatial_statistics_t* s, int32_t code) {
    for (int32_t i = 0; i < s->num_types; i++) {
        if (s->types[i] == code) return;
    }
    if (s->num_types < CARQUET_GEO_MAX_TYPES) {
        s->types[s->num_types++] = code;
    }
}

/* The specification skips null and NaN values per dimension: POINT (1 NaN)
 * contributes 1 to X and nothing to Y. A box exists once both X and Y have a
 * value. Infinities are values, and are kept. */
static void fold(double v, bool* seen, double* lo, double* hi) {
    if (isnan(v)) return;
    if (!*seen) { *lo = *hi = v; *seen = true; return; }
    if (v < *lo) *lo = v;
    if (v > *hi) *hi = v;
}

static void add_coord(parquet_geospatial_statistics_t* s,
                      double x, double y, int has_z, double z,
                      int has_m, double m) {
    if (s->omit_bbox) return;
    fold(x, &s->has_x, &s->xmin, &s->xmax);
    fold(y, &s->has_y, &s->ymin, &s->ymax);
    s->valid = s->has_x && s->has_y;
    if (has_z) fold(z, &s->has_z, &s->zmin, &s->zmax);
    if (has_m) fold(m, &s->has_m, &s->mmin, &s->mmax);
}

typedef struct {
    const uint8_t* p;
    size_t n;
    size_t off;
    int bad;
} cur_t;

static uint32_t rd_u32(cur_t* c, int le) {
    if (c->bad || c->off + 4 > c->n) { c->bad = 1; return 0; }
    const uint8_t* b = c->p + c->off;
    c->off += 4;
    return le ? ((uint32_t)b[0] | ((uint32_t)b[1] << 8) |
                 ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24))
              : ((uint32_t)b[3] | ((uint32_t)b[2] << 8) |
                 ((uint32_t)b[1] << 16) | ((uint32_t)b[0] << 24));
}

static double rd_f64(cur_t* c, int le) {
    if (c->bad || c->off + 8 > c->n) { c->bad = 1; return 0.0; }
    uint8_t t[8];
    if (le) memcpy(t, c->p + c->off, 8);
    else for (int i = 0; i < 8; i++) t[i] = c->p[c->off + 7 - i];
    c->off += 8;
    double d;
    memcpy(&d, t, 8);
    return d;
}

static void read_points(parquet_geospatial_statistics_t* s, cur_t* c, int le,
                         uint32_t count, int ndim, int hz, int hm) {
    for (uint32_t i = 0; i < count && !c->bad; i++) {
        double v[4] = {0,0,0,0};
        for (int d = 0; d < ndim; d++) v[d] = rd_f64(c, le);
        if (c->bad) return;
        double z = hz ? v[2] : 0.0;
        double m = hm ? v[hz ? 3 : 2] : 0.0;
        add_coord(s, v[0], v[1], hz, z, hm, m);
    }
}

static void walk(parquet_geospatial_statistics_t* s, cur_t* c, int depth) {
    if (c->bad || depth > 32) { c->bad = 1; return; }

    if (c->off + 1 > c->n) { c->bad = 1; return; }
    int le = c->p[c->off] == 1;
    c->off += 1;

    uint32_t raw = rd_u32(c, le);
    if (c->bad) return;

    int hz, hm, base;
    if (raw & 0xE0000000u) {                 /* EWKB (PostGIS) flags */
        hz = (raw & 0x80000000u) != 0;
        hm = (raw & 0x40000000u) != 0;
        int srid = (raw & 0x20000000u) != 0;
        base = (int)(raw & 0xFFu);
        if (srid) { (void)rd_u32(c, le); if (c->bad) return; }
    } else {                                  /* ISO WKB */
        base = (int)(raw % 1000u);
        unsigned d = raw / 1000u;
        hz = (d == 1 || d == 3);
        hm = (d == 2 || d == 3);
    }
    int ndim = 2 + hz + hm;
    int32_t iso = (int32_t)base + (hz ? 1000 : 0) + (hm ? 2000 : 0);
    /* The types are those of the column's values, not of their parts: a
     * MultiPolygon is a 6, not {6, 3}, and a collection is a 7. */
    if (depth == 0) add_type(s, iso);

    switch (base) {
    case 1:  /* Point */
        read_points(s, c, le, 1, ndim, hz, hm);
        break;
    case 2: { /* LineString */
        uint32_t n = rd_u32(c, le);
        read_points(s, c, le, n, ndim, hz, hm);
        break;
    }
    case 3: { /* Polygon */
        uint32_t rings = rd_u32(c, le);
        for (uint32_t r = 0; r < rings && !c->bad; r++) {
            uint32_t npts = rd_u32(c, le);
            read_points(s, c, le, npts, ndim, hz, hm);
        }
        break;
    }
    case 4:  /* MultiPoint */
    case 5:  /* MultiLineString */
    case 6:  /* MultiPolygon */
    case 7: { /* GeometryCollection */
        uint32_t n = rd_u32(c, le);
        for (uint32_t i = 0; i < n && !c->bad; i++) walk(s, c, depth + 1);
        break;
    }
    default:
        c->bad = 1;       /* unknown geometry type: stop */
        break;
    }
}

void carquet_geo_stats_add_wkb(parquet_geospatial_statistics_t* s,
                               const uint8_t* wkb, size_t len) {
    if (!s || !wkb || len < 5) return;
    cur_t c = { wkb, len, 0, 0 };
    walk(s, &c, 0);
}

void carquet_geo_stats_merge(parquet_geospatial_statistics_t* dst,
                             const parquet_geospatial_statistics_t* src) {
    if (!src->has_x && !src->has_y && !src->has_z && !src->has_m &&
        src->num_types == 0) {
        return;
    }
    if (!dst->omit_bbox) {
        if (src->has_x) {
            fold(src->xmin, &dst->has_x, &dst->xmin, &dst->xmax);
            fold(src->xmax, &dst->has_x, &dst->xmin, &dst->xmax);
        }
        if (src->has_y) {
            fold(src->ymin, &dst->has_y, &dst->ymin, &dst->ymax);
            fold(src->ymax, &dst->has_y, &dst->ymin, &dst->ymax);
        }
        dst->valid = dst->has_x && dst->has_y;
        if (src->has_z) {
            fold(src->zmin, &dst->has_z, &dst->zmin, &dst->zmax);
            fold(src->zmax, &dst->has_z, &dst->zmin, &dst->zmax);
        }
        if (src->has_m) {
            fold(src->mmin, &dst->has_m, &dst->mmin, &dst->mmax);
            fold(src->mmax, &dst->has_m, &dst->mmin, &dst->mmax);
        }
    }
    for (int32_t i = 0; i < src->num_types; i++) add_type(dst, src->types[i]);
}
