#pragma once
#include "Base.hpp"
#include <cstddef>
#include <memory>

namespace kun {

namespace detail {
struct SkipListImpl;
}

struct OutputStreamBase;
struct InputStreamBase;


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
    double get(int rank, size_t &index, bool &found);
    int size() const;
    bool serialize(OutputStreamBase *stream, int expsize) const;
    bool deserialize(InputStreamBase *stream, int expsize);
};

KUN_API bool serializeSkipList(SkipList *skiplist, int *lastInsertRank, size_t size,
                       size_t window, OutputStreamBase *stream);
KUN_API bool deserializeSkipList(/*uninitialized*/ SkipList *skiplist,
                         int *lastInsertRank, size_t size, size_t window,
                         InputStreamBase *stream);

} // namespace kun