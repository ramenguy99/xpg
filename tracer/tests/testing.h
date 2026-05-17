#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#ifndef _WIN32
#include <signal.h>
#endif

static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;
static int g_current_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN(name) do { \
    g_current_failed = 0; \
    printf("  %-55s ", #name); \
    fflush(stdout); \
    test_##name(); \
    g_tests_run++; \
    if (g_current_failed) { g_tests_failed++; } \
    else { g_tests_passed++; printf("PASS\n"); } \
} while (0)

#define CHECK(cond) do { \
    if (!(cond)) { \
        if (!g_current_failed) printf("FAIL\n"); \
        fprintf(stderr, "    %s:%d: CHECK( %s )\n", __FILE__, __LINE__, #cond); \
        g_current_failed = 1; \
        return; \
    } \
} while (0)

#define CHECK_EQ(a, b) do { \
    long long _a = (long long)(a), _b = (long long)(b); \
    if (_a != _b) { \
        if (!g_current_failed) printf("FAIL\n"); \
        fprintf(stderr, "    %s:%d: CHECK_EQ( %s, %s )  =>  %lld != %lld\n", \
                __FILE__, __LINE__, #a, #b, _a, _b); \
        g_current_failed = 1; \
        return; \
    } \
} while (0)

#ifndef _WIN32
static void _test_timeout_handler(int sig) {
    (void)sig;
    fprintf(stderr, "\nTEST TIMEOUT -- possible deadlock\n");
    _exit(2);
}
#endif

static inline void test_setup_timeout(int seconds) {
#ifndef _WIN32
    signal(SIGALRM, _test_timeout_handler);
    alarm(seconds);
#else
    (void)seconds;
#endif
}

static inline int test_print_results(const char* suite_name) {
    printf("\n=== %s: %d/%d passed", suite_name, g_tests_passed, g_tests_run);
    if (g_tests_failed > 0)
        printf(", %d FAILED", g_tests_failed);
    printf(" ===\n");
    return g_tests_failed > 0 ? 1 : 0;
}
