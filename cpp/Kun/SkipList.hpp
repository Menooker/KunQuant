#pragma once
#include "Base.hpp"
#include <cstddef>
#include <memory>

namespace kun {

namespace detail {
    struct SkipListImpl;
}

struct KUN_API SkipList {
    std::unique_ptr<detail::SkipListImpl> impl;
    SkipList(int size);
    SkipList();
    void init(int size);
    ~SkipList();
    // Returns the rank of the inserted element. When there are duplicates
    // `rank` is the highest of the group, i.e. the 'max' method of
    // https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rank.html
    int insert(double value, size_t index);
    // remove the first inserted element with the given value
    bool remove(double value);
    int minRank(double value);
    double get(int rank, size_t& index, bool& found);
    int size() const;
};

}