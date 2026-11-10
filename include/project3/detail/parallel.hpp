#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace project3::detail {

[[nodiscard]] inline std::size_t resolve_worker_count(
    const std::size_t requested) {
  if (requested == 1) return 1;
  const auto hardware = static_cast<std::size_t>(
      std::max(1u, std::thread::hardware_concurrency()));
  if (requested == 0) return hardware;
  return std::min(requested, hardware);
}

/// Independent index-space map. Work items must not share mutable state.
template <typename Fn>
void parallel_for(const std::size_t count,
                  const std::size_t requested_workers,
                  Fn fn) {
  if (count == 0) return;
  const auto workers = std::min(resolve_worker_count(requested_workers), count);
  if (workers == 1) {
    for (std::size_t i = 0; i < count; ++i) fn(i);
    return;
  }
  std::atomic<std::size_t> next{0};
  std::mutex error_mutex;
  std::exception_ptr error;
  std::vector<std::thread> threads;
  threads.reserve(workers);
  for (std::size_t t = 0; t < workers; ++t) {
    threads.emplace_back([&] {
      try {
        while (true) {
          const auto index = next.fetch_add(1, std::memory_order_relaxed);
          if (index >= count) return;
          fn(index);
        }
      } catch (...) {
        std::lock_guard lock{error_mutex};
        if (!error) error = std::current_exception();
      }
    });
  }
  for (auto& thread : threads) thread.join();
  if (error) std::rethrow_exception(error);
}

/// Persistent worker pool for repeated small parallel sections. Unlike
/// parallel_for it does not create threads per call, so it pays off when the
/// same index-space map runs thousands of times (for example, one gradient
/// evaluation per optimizer iteration). The calling thread is participant 0
/// and pool thread i is participant i+1; each participant processes one
/// contiguous chunk, so results do not depend on scheduling. Jobs must not
/// call back into the pool.
class ChunkedPool final {
 public:
  static ChunkedPool& instance() {
    static ChunkedPool pool;
    return pool;
  }

  template <typename Fn>
  void run(const std::size_t count, const std::size_t requested_workers,
           Fn&& fn) {
    if (count == 0) return;
    const auto hardware = resolve_worker_count(0);
    const auto requested =
        requested_workers == 0 ? hardware
                               : std::min(requested_workers, hardware);
    const auto participants =
        std::max<std::size_t>(1, std::min({requested, count, hardware}));
    if (participants <= 1) {
      for (std::size_t i = 0; i < count; ++i) fn(i);
      return;
    }
    std::function<void(std::size_t)> job = std::forward<Fn>(fn);
    const std::size_t chunk = (count + participants - 1) / participants;
    {
      std::lock_guard lock{mutex_};
      job_ = &job;
      count_ = count;
      chunk_ = chunk;
      participants_ = participants;
      remaining_ = participants - 1;
      ++job_id_;
    }
    cv_.notify_all();
    process_chunk(0);
    {
      std::unique_lock lock{mutex_};
      done_cv_.wait(lock, [&] { return remaining_ == 0; });
      job_ = nullptr;
    }
    if (error_) std::rethrow_exception(error_);
  }

 private:
  ChunkedPool() {
    const auto workers = resolve_worker_count(0);
    threads_.reserve(workers > 0 ? workers - 1 : 0);
    for (std::size_t i = 1; i < workers; ++i) {
      threads_.emplace_back([this, slot = i] { worker_loop(slot); });
    }
  }
  ~ChunkedPool() {
    {
      std::lock_guard lock{mutex_};
      stopping_ = true;
    }
    cv_.notify_all();
    for (auto& thread : threads_) thread.join();
  }
  ChunkedPool(const ChunkedPool&) = delete;
  ChunkedPool& operator=(const ChunkedPool&) = delete;

  void process_chunk(const std::size_t participant) {
    const std::size_t first = participant * chunk_;
    if (first >= count_) return;
    const std::size_t last = std::min(count_, first + chunk_);
    try {
      for (std::size_t i = first; i < last; ++i) (*job_)(i);
    } catch (...) {
      std::lock_guard lock{mutex_};
      if (!error_) error_ = std::current_exception();
    }
  }

  void worker_loop(const std::size_t slot) {
    std::uint64_t seen = 0;
    for (;;) {
      std::uint64_t id = 0;
      std::size_t participants = 0;
      {
        std::unique_lock lock{mutex_};
        cv_.wait(lock, [&] { return stopping_ || job_id_ != seen; });
        if (stopping_) return;
        id = job_id_;
        participants = participants_;
      }
      if (slot < participants) {
        process_chunk(slot);
        {
          std::lock_guard lock{mutex_};
          if (remaining_ > 0) --remaining_;
          done_cv_.notify_all();
        }
      }
      seen = id;
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  std::function<void(std::size_t)>* job_{};
  std::size_t count_{};
  std::size_t chunk_{};
  std::size_t participants_{};
  std::size_t remaining_{};
  std::uint64_t job_id_{};
  bool stopping_{};
  std::exception_ptr error_;
  std::vector<std::thread> threads_;
};

}  // namespace project3::detail
