#ifndef _RKNN_DEMO_TIMER_H_
#define _RKNN_DEMO_TIMER_H_

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Define this macro to disable timing logs
// #define TIMING_DISABLED // if you don't need to print the time used, uncomment this line of code

class TIMER
{
private:
    struct timeval start_time, stop_time;
    double get_microseconds(struct timeval t) { return (t.tv_sec * 1000000.0 + t.tv_usec); }
    char indent[40] = {0};  // Initialize all bytes to zero
    bool initialized = false;

public:
    TIMER() {}
    // Disable copy constructor and copy assignment to prevent shallow copies
    TIMER(const TIMER&) = delete;
    TIMER& operator=(const TIMER&) = delete;
    virtual ~TIMER() = default;  // Virtual destructor for proper inheritance support

    /**
     * @brief Set indentation prefix for timing logs
     * @param s Indentation string (NULL or empty sets default "-- ")
     * @note Empty string (s[0]=='\0') triggers default "-- " for safety
     */
    void indent_set(const char *s)
    {
        if (s == NULL || s[0] == '\0') {
            // Handle NULL or empty string: use default "-- "
            memset(indent, 0, sizeof(indent));
            strncpy(indent, "-- ", sizeof(indent) - 1);
        } else {
            strncpy(indent, s, sizeof(indent) - 1);
        }
        indent[sizeof(indent) - 1] = '\0';
    }

    void tik()
    {
        gettimeofday(&start_time, NULL);
        initialized = true;
    }

    void tok()
    {
        if (!initialized) {
            fprintf(stderr, "Error: TIMER::tok() called before TIMER::tik(). Time measurement will be invalid.\n");
            gettimeofday(&start_time, NULL);  // Initialize to current time to prevent negative values
            gettimeofday(&stop_time, NULL);   // Set stop time equal to start time
            initialized = true;
            return;
        }
        gettimeofday(&stop_time, NULL);
    }

#ifdef TIMING_DISABLED
    void print_time(const char *str)
    {
        // No action if TIMING_DISABLED is defined
    }
#else
    void print_time(const char *str)
    {
        printf("%s", indent);
        printf("%s use: %f ms\n", str, get_time());
    }
#endif

    float get_time()
    {
        if (!initialized) {
            fprintf(stderr, "Error: TIMER::get_time() called before TIMER::tik() and TIMER::tok()\n");
            return 0.0f;
        }
        return (get_microseconds(stop_time) - get_microseconds(start_time)) / 1000.0;
    }
};

#endif // _RKNN_DEMO_TIMER_H_
