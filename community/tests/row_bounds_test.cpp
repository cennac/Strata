#include "strata/core/batch_rows.hpp"
#include <cassert>
#include <cstdio>

int main() {
    // Two 2-slot groups, two proposals each: group 1 occupies rows [6,12).
    int rows[] = {2,2,2,3,3,3};
    int64_t pos[] = {130030,130031,130032,129900,129901,129902};
    using strata::core::batch_rows_error;
    assert(batch_rows_error(rows,6,pos,4,6,6,true,12)==nullptr);
    assert(batch_rows_error(rows,6,pos,4,6,6,true,8)!=nullptr);
    assert(batch_rows_error(rows,6,pos,4,6,7,true,12)!=nullptr);
    assert(batch_rows_error(rows,6,pos,4,5,6,true,12)!=nullptr);
    pos[2]++;
    assert(batch_rows_error(rows,6,pos,4,6,6,true,12)!=nullptr);
    pos[2]--;
    rows[5]=2;
    assert(batch_rows_error(rows,6,pos,4,6,6,true,12)!=nullptr);
    rows[5]=4;
    assert(batch_rows_error(rows,6,pos,4,6,6,true,12)!=nullptr);
    // Two internal slots: one slot per group, three rows each, fits the old extent.
    int narrow[] = {1,1,1};
    int64_t p[] = {130030,130031,130032};
    assert(batch_rows_error(narrow,3,p,2,3,3,true,8)==nullptr);
    std::puts("pipelined MTP row bounds passed");
}
