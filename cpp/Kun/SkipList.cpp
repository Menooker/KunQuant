/*
Modified from
https://github.com/pandas-dev/pandas/blob/main/pandas/_libs/include/pandas/skiplist.h
Original copyright:
Copyright (c) 2016, PyData Development Team
All rights reserved.

Distributed under the terms of the BSD Simplified License.

The full license is in the LICENSE file, distributed with this software.

Flexibly-sized, index-able skiplist data structure for maintaining a sorted
list of values

Port of Wes McKinney's Cython version of Raymond Hettinger's original pure
Python recipe (https://rhettinger.wordpress.com/2010/02/06/lost-knowledge/)
*/

#include "SkipList.hpp"
#include "StateBuffer.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <vector>
namespace kun {

namespace detail {
template <typename T>
class RcPtr {
    T *ptr;
    void deref() {
        if (ptr) {
            ptr->ref_count--;
            if (ptr->ref_count == 0) {
                delete ptr;
            }
        }
    }

  public:
    explicit RcPtr(T *ptr) noexcept : ptr{ptr} {}
    RcPtr() noexcept : ptr{nullptr} {}
    ~RcPtr() { deref(); }
    RcPtr(const RcPtr &other) noexcept : ptr{other.ptr} {
        if (ptr) {
            ptr->ref_count++;
        }
    }
    RcPtr(RcPtr &&other) noexcept : ptr{other.ptr} { other.ptr = nullptr; }
    RcPtr &operator=(std::nullptr_t) noexcept {
        deref();
        ptr = nullptr;
        return *this;
    }
    RcPtr &operator=(RcPtr &&other) noexcept {
        if (this == &other) {
            return *this;
        }
        deref();
        ptr = other.ptr;
        other.ptr = nullptr;
        return *this;
    }
    RcPtr &operator=(const RcPtr &other) noexcept {
        if (this == &other) {
            return *this;
        }
        deref();
        ptr = other.ptr;
        if (ptr) {
            ptr->ref_count++;
        }
        return *this;
    }
    T *operator->() const { return ptr; }
    T &operator*() const { return *ptr; }
    bool operator==(const RcPtr &other) const { return ptr == other.ptr; }
    bool operator!=(const RcPtr &other) const { return ptr != other.ptr; }
    T *get() const { return ptr; }
};

static inline float __skiplist_nanf(void) {
    const union {
        int __i;
        float __f;
    } __bint = {0x7fc00000UL};
    return __bint.__f;
}
#define PANDAS_NAN ((double)__skiplist_nanf())

static inline double Log2(double val) { return std::log(val) / std::log(2.); }

static inline double urand(void) {
    return ((double)rand() + 1) / ((double)RAND_MAX + 2);
}

static inline int int_min(int a, int b) { return a < b ? a : b; }

struct Node {
    std::vector<RcPtr<Node>> next;
    std::unique_ptr<int[]> width;
    double value;
    size_t index;
    int is_nil;
    int getLevels() const { return next.size(); }
    int ref_count;

    Node(double value, size_t index, int levels)
        : next(levels), width(new int[levels]), value(value), index(index),
          is_nil(0), ref_count(1) {}

    // 1 if left < right, 0 if left == right, -1 if left > right
    int cmp(double value) const {
        if (is_nil || this->value > value) {
            return -1;
        } else if (this->value < value) {
            return 1;
        } else {
            return 0;
        }
    }
};

struct SkipListImpl {
    RcPtr<Node> head;
    std::vector<Node *> tmp_chain;
    std::unique_ptr<int[]> tmp_steps;
    int size;
    int getMaxLevels() const { return tmp_chain.size(); };
    SkipListImpl(int size) {
        SkipListImpl *result = this;
        int maxlevels, i;

        maxlevels = 1 + Log2((double)size);
        result->tmp_chain = std::vector<Node *>(maxlevels);
        result->tmp_steps = std::unique_ptr<int[]>(new int[maxlevels]);
        result->size = 0;

        head = RcPtr<Node>(new Node(PANDAS_NAN, 0, maxlevels));
        RcPtr<Node> NIL{new Node(0.0, 0, 0)};
        NIL->is_nil = 1;

        for (i = 0; i < maxlevels; ++i) {
            head->next[i] = NIL;
            head->width[i] = 1;
        }
    }

    double get(int i, size_t &index, bool &ret) const {
        int level;

        if (i < 0 || i >= size) {
            ret = false;
            return 0;
        }

        Node *node = head.get();
        ++i;
        for (int level = getMaxLevels() - 1; level >= 0; --level) {
            while (node->width[level] <= i) {
                i -= node->width[level];
                node = node->next[level].get();
            }
        }

        ret = true;
        index = node->index;
        return node->value;
    }

    // Returns the lowest rank of all elements with value `value`, as opposed to
    // the
    // highest rank returned by `skiplist_insert`.
    int minRank(double value) const {
        int rank = 0;

        Node *node = head.get();
        for (int level = getMaxLevels() - 1; level >= 0; --level) {
            while (node->next[level]->cmp(value) > 0) {
                rank += node->width[level];
                node = node->next[level].get();
            }
        }

        return rank;
    }

    // Returns the rank of the inserted element. When there are duplicates,
    // `rank` is the highest of the group, i.e. the 'max' method of
    // https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rank.html
    int insert(double value, size_t index) {
        // Node *node, *prevnode, *newnode, *next_at_level;
        int *steps_at_level = tmp_steps.get();
        // int size, steps, level, rank = 0;
        int rank = 0;

        memset(steps_at_level, 0, getMaxLevels() * sizeof(int));

        Node *node = head.get();

        for (int level = getMaxLevels() - 1; level >= 0; --level) {
            Node *next_at_level = node->next[level].get();
            while (next_at_level->cmp(value) >= 0) {
                steps_at_level[level] += node->width[level];
                rank += node->width[level];
                node = next_at_level;
                next_at_level = node->next[level].get();
            }
            tmp_chain[level] = node;
        }

        int size = int_min(getMaxLevels(), 1 - ((int)Log2(urand())));

        RcPtr<Node> newnode{new Node(value, index, size)};
        int steps = 0;

        for (int level = 0; level < size; ++level) {
            auto prevnode = tmp_chain[level];
            newnode->next[level] = prevnode->next[level];
            newnode->width[level] = prevnode->width[level] - steps;
            prevnode->next[level] = newnode;
            prevnode->width[level] = steps + 1;

            steps += steps_at_level[level];
        }

        for (int level = size; level < getMaxLevels(); ++level) {
            tmp_chain[level]->width[level] += 1;
        }

        ++(this->size);

        return rank;
    }

    bool remove(double value) {
        Node *node = head.get();

        for (int level = getMaxLevels() - 1; level >= 0; --level) {
            Node *next_at_level = node->next[level].get();
            while (next_at_level->cmp(value) > 0) {
                node = next_at_level;
                next_at_level = node->next[level].get();
            }
            tmp_chain[level] = node;
        }

        if (value != tmp_chain[0]->next[0]->value) {
            return false;
        }

        int size = tmp_chain[0]->next[0]->getLevels();

        for (int level = 0; level < size; ++level) {
            Node *prevnode = tmp_chain[level];

            RcPtr<Node> tmpnode = prevnode->next[level];

            prevnode->width[level] += tmpnode->width[level] - 1;
            prevnode->next[level] = tmpnode->next[level];

            tmpnode->next[level] = nullptr;
        }

        for (int level = size; level < getMaxLevels(); ++level) {
            --(tmp_chain[level]->width[level]);
        }

        --(this->size);
        return true;
    }
};

} // namespace detail

SkipList::SkipList(int size) : impl{new detail::SkipListImpl(size)} {}
SkipList::SkipList() = default;
void SkipList::init(int size) {
    impl = std::unique_ptr<detail::SkipListImpl>(new detail::SkipListImpl(size));
}

SkipList::~SkipList() = default;

int SkipList::insert(double value, size_t index) {
    return impl->insert(value, index);
}

bool SkipList::remove(double value) { return impl->remove(value); }

int SkipList::minRank(double value) { return impl->minRank(value); }

double SkipList::get(int rank, size_t &index, bool &found) {
    return impl->get(rank, index, found);
}

int SkipList::size() const { return impl->size; }

bool SkipList::serialize(OutputStreamBase *stream, int expsize) const {
    if (!stream->write(&impl->size, sizeof(impl->size))) {
        return false;
    }
    for(int i=0;i<impl->size;i++) {
        size_t index;
        bool found;
        double value = impl->get(i, index, found);
        if (!found) {
            return false;
        }
        if (!stream->write(&value, sizeof(value))) {
            return false;
        }
        if (!stream->write(&index, sizeof(index))) {
            return false;
        }
    }
    return true;
}

bool SkipList::deserialize(InputStreamBase *stream, int expsize) {
    size_t size;
    if (!stream->read(&size, sizeof(size))) {
        return false;
    }
    for (size_t i = 0; i < size; i++) {
        double value;
        size_t index;
        if (!stream->read(&value, sizeof(value))) {
            return false;
        }
        if (!stream->read(&index, sizeof(index))) {
            return false;
        }
        impl->insert(value, index);
    }
    return true;
}


bool serializeSkipList(SkipList* skiplist, int* lastInsertRank, size_t size, size_t window, OutputStreamBase* stream) {
    if (!stream->write(lastInsertRank, size * sizeof(int))) {
        return false;
    }
    for (size_t i = 0; i < size; i++) {
        if (!skiplist->serialize(stream, window)) {
            return false;
        }
    }
    return true;
}
bool deserializeSkipList(SkipList* skiplist, int* lastInsertRank, size_t size, size_t window, InputStreamBase* stream) {
    if (!stream->read(lastInsertRank, size * sizeof(int))) {
        return false;
    }
    for (size_t i = 0; i < size; i++) {
        if (!skiplist[i].deserialize(stream, window)) {
            return false;
        }
    }
    return true;
}
} // namespace kun