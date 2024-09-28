#pragma once

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>
#include <ctime>

// ANSI color codes
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_RESET   "\x1b[0m"

template<typename T>
int cmpArrays(int n, T *a, T *b) {
    for (int i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            printf("    a[%d] = %d, b[%d] = %d\n", i, a[i], i, b[i]);
            return 1;
        }
    }
    return 0;
}

void printDesc(const char *desc) {
    printf("==== %s ====\n", desc);
}

template<typename T>
void printCmpResult(int n, T* a, T* b) {
    if (cmpArrays(n, a, b)) {
        std::cout << "    " << ANSI_COLOR_RED << "FAIL VALUE" << ANSI_COLOR_RESET << std::endl;
    }
    else {
        std::cout << "    " << ANSI_COLOR_GREEN << "passed" << ANSI_COLOR_RESET << std::endl;
    }
}

template<typename T>
void printCmpLenResult(int n, int expN, T* a, T* b) {
    if (n != expN) {
        std::cout << "    expected " << expN << " elements, got " << n << std::endl;
    }

    if (n == -1 || n != expN) {
        std::cout << "    " << ANSI_COLOR_RED << "FAIL COUNT" << ANSI_COLOR_RESET << std::endl;
    }
    else if (cmpArrays(n, a, b)) {
        std::cout << "    " << ANSI_COLOR_RED << "FAIL VALUE" << ANSI_COLOR_RESET << std::endl;
    }
    else {
        std::cout << "    " << ANSI_COLOR_GREEN << "passed" << ANSI_COLOR_RESET << std::endl;
    }
}

void zeroArray(int n, int *a) {
    for (int i = 0; i < n; i++) {
        a[i] = 0;
    }
}

void onesArray(int n, int *a) {
    for (int i = 0; i < n; i++) {
        a[i] = 1;
    }
}

void genArray(int n, int *a, int maxval) {
    srand(time(nullptr));

    for (int i = 0; i < n; i++) {
        a[i] = rand() % maxval;
    }
}

void printArray(int n, int *a, bool abridged = false) {
    printf("    [ ");
    for (int i = 0; i < n; i++) {
        if (abridged && i + 2 == 15 && n > 16) {
            i = n - 2;
            printf("... ");
        }
        printf("%3d ", a[i]);
    }
    printf("]\n");
}

template<typename T>
void printElapsedTime(T time, std::string note = "")
{
    std::cout << "   elapsed time: " << time << "ms    " << note << std::endl;
}
