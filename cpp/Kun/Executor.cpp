#include "Context.hpp"
#include "Module.hpp"
#include <algorithm>
#include <array>
#include <assert.h>
#include <condition_variable>
#include <cstdio>
#include <list>
#include <mutex>
#include <thread>

namespace kun {

struct SingleThreadExecutor : Executor {
    std::list<RuntimeStage *> q;
    virtual void enqueue(RuntimeStage *stage) override { q.push_front(stage); }

    virtual void dequeue(RuntimeStage *stage) override {
        q.erase(std::find(q.begin(), q.end(), stage));
    }

    bool takeSingleJob() {
        if (q.empty()) {
            return false;
        }
        q.front()->doJob();
        return true;
    }

    void runUntilDone() override {
        while (takeSingleJob()) {
            /* code */
        }
    }

    ~SingleThreadExecutor() = default;
};

std::shared_ptr<Executor> createSingleThreadExecutor() {
    return std::make_shared<SingleThreadExecutor>();
}

struct MultiThreadExecutor : Executor {
    std::mutex main_lock;
    std::mutex qlock;
    std::vector<std::thread> threads;
    std::vector<RuntimeStage *> q;
    std::array<std::atomic<RuntimeStage *>, 4> fast_slots;
    std::atomic<size_t> num_stages{0};

    std::condition_variable cv;
    std::mutex cv_lock;
    std::atomic<size_t> idle_count{0};
    bool closing = false;

    void notifyAwaiters() { cv.notify_all(); }

    void park(int &count) {
        count++;
        if (count > 20 || num_stages.load() == 0) {
            ++idle_count;
            count = 0;
            std::unique_lock<std::mutex> lk{cv_lock};
            cv.wait(lk);
            --idle_count;
        }
    }

    MultiThreadExecutor(int num_threads) {
        for (auto &slot : fast_slots) {
            slot.store(nullptr);
        }
        q.reserve(64);
        threads.reserve(num_threads);
        for (int i = 0; i < num_threads; i++) {
            threads.emplace_back([this, i]() { workerMain(i); });
        }
    }

    virtual void enqueue(RuntimeStage *stage) override {
        ++num_stages;
        for (auto &slot : fast_slots) {
            auto curstage = slot.load();
            if (!curstage) {
                if (slot.compare_exchange_strong(curstage, stage)) {
                    notifyAwaiters();
                    return;
                }
            }
        }
        {
            std::lock_guard<std::mutex> guard{qlock};
            q.push_back(stage);
        }
        notifyAwaiters();
    }

    virtual void dequeue(RuntimeStage *stage) override {
        // important! After --num_stages, do not Read RuntimeStage. This also
        // apply on functions of RuntimeStage after calling dequeue. Otherwise,
        // we may read the RuntimeStage object of the next iteration
        for (auto &slot : fast_slots) {
            auto curstage = slot.load();
            if (curstage == stage) {
                slot.store(nullptr);
                --num_stages;
                return;
            }
        }
        std::lock_guard<std::mutex> guard{qlock};
        auto itr = std::find(q.begin(), q.end(), stage);
        assert(itr != q.end());
        q.erase(itr);
        --num_stages;
    }

    RuntimeStage *takeSingleJob() {
        for (auto &slot : fast_slots) {
            auto curstage = slot.load();
            if (curstage && curstage->hasJobToDo()) {
                return curstage;
            }
        }
        RuntimeStage *stage = nullptr;
        {
            std::lock_guard<std::mutex> guard{qlock};
            if (q.empty()) {
                return nullptr;
            }
            for (auto itr = q.rbegin(); itr != q.rend(); ++itr) {
                auto cur = *itr;
                if (cur->hasJobToDo()) {
                    stage = cur;
                    break;
                }
            }
        }
        return stage;
    }

    RuntimeStage *workerTakeJob() {
        if (closing) {
            return nullptr;
        }
        if (num_stages.load() <= 0) {
            return nullptr;
        }
        return takeSingleJob();
    }

    void workerMain(int tid) {
        int parkcount = 0;
        for (;;) {
            auto job = workerTakeJob();
            while (job) {
                job->doJob();
                job = workerTakeJob();
            }
            // printf("PARK %d\n", tid);
            park(parkcount);
            if (closing) {
                return;
            }
        }
    }

    void runUntilDone() override {
        std::lock_guard<std::mutex> guard{main_lock};
        while (num_stages.load() > 0) {
            auto job = takeSingleJob();
            if (!job) {
                continue;
            }
            job->doJob();
        }
        while (idle_count != threads.size()) {
        }
    }

    ~MultiThreadExecutor() {
        {
            std::unique_lock<std::mutex> lk{cv_lock};
            closing = true;
        }
        notifyAwaiters();
        for (auto &t : threads) {
            t.join();
        }
    }
};

std::shared_ptr<Executor> createMultiThreadExecutor(int num_threads) {
    return std::make_shared<MultiThreadExecutor>(num_threads);
}

} // namespace kun