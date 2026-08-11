
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

enum TokenizerLogLevel
{
    /*1*/
    DBG_FETAL = 1 << 0,
    /*2*/
    DBG_ERROR = 1 << 1,
    /*4*/
    DBG_WARN = 1 << 2,
    /*8*/
    DBG_INFO = 1 << 3,
    /*16*/
    DBG_DEBUG = 1 << 4,
    /*32*/
    DBG_VERBOSE = 1 << 5,
    /*Mask*/
    DBG_MARSK = 0xFF,
};

#define LOG(levelStr, _str, ...)                                                 \
    do                                                                        \
    {                                                                         \
        fprintf(stdout, levelStr LOG_TAG ": %s,line=%d, " _str, __FUNCTION__, __LINE__, \
                ##__VA_ARGS__);                                               \
        fprintf(stdout, "\n");                                                \
    } while (0)


#define LOG_D(_str, ...)    LOG("D ", _str, ##__VA_ARGS__)
#define LOG_I(_str, ...)    LOG("I ", _str, ##__VA_ARGS__)
#define LOG_W(_str, ...)    LOG("W ", _str, ##__VA_ARGS__)
#define LOG_E(_str, ...)    LOG("E ", _str, ##__VA_ARGS__)
