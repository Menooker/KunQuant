#include <Kun/SkipList.hpp>
#include <stdio.h>

#define CHECK(V)                                                               \
    if (!(V)) {                                                                \
        printf("CHECK(" #V ") at " __FILE__ ":%d failed\n", __LINE__);         \
        return false;                                                          \
    }

bool testSkipList() {
    kun::SkipList list(16);
    list.insert(0, 0);
    list.insert(0, 1);
    for (int i = 2; i < 16; i++) {
        list.insert(i, i);
    }
    // (0, 0), (0, 1), (2, 2), (3, 3), (4, 4), ..., (15, 15)
    size_t index;
    bool found;
    auto result = list.get(0, index, found);
    CHECK(found);
    CHECK(result == 0);
    CHECK(index == 0);

    // (0, 1), (2, 2), (3, 3), (4, 4), ..., (15, 15)
    CHECK(list.remove(0));
    result = list.get(0, index, found);
    CHECK(found);
    CHECK(result == 0);
    CHECK(index == 1);

    // (0, 1), (0, 16), (0, 17), (2, 2), (3, 3), (4, 4), ..., (15, 15)
    CHECK(list.insert(0, 16) == 1);
    CHECK(list.insert(0, 17) == 2);
    result = list.get(0, index, found);
    CHECK(found);
    CHECK(result == 0);
    CHECK(index == 1);

    // (0, 16), (0, 17), (2, 2), (3, 3), (4, 4), ..., (15, 15)
    CHECK(list.remove(0));
    result = list.get(0, index, found);
    CHECK(found);
    CHECK(result == 0);
    CHECK(index == 16);

    auto rank = list.minRank(0);
    CHECK(rank == 0);

    // (0, 16), (0, 17), (2, 2), ..., (13, 13), (13, 18), (14, 14), ..., (15,
    // 15)
    CHECK(list.insert(13, 18) == 14);
    CHECK(list.minRank(13) == 13);
    result = list.get(13, index, found);
    CHECK(found);
    CHECK(result == 13);
    CHECK(index == 13);
    result = list.get(14, index, found);
    CHECK(found);
    CHECK(result == 13);
    CHECK(index == 18);
    result = list.get(15, index, found);
    CHECK(found);
    CHECK(result == 14);
    CHECK(index == 14);

    return true;
}