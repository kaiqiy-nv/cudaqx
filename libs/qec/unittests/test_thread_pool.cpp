/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/// @file
///
/// Unit tests for the shared persistent QEC thread pool.

#include "../lib/decoding_task/thread_pool.h"
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>

namespace cudaq::qec::detail {
namespace {

TEST(QecThreadPool, ParallelForExecutesEveryTaskAndCanBeReused) {
  thread_pool pool(/*num_threads=*/4);
  std::array<std::atomic<std::size_t>, 32> executions{};
  std::array<std::atomic<std::size_t>, 32> workers{};

  pool.parallel_for(32, [&](std::size_t task, std::size_t worker) {
    executions[task].fetch_add(1, std::memory_order_relaxed);
    workers[task].store(worker, std::memory_order_relaxed);
  });
  for (std::size_t task = 0; task < executions.size(); ++task) {
    EXPECT_EQ(executions[task].load(std::memory_order_relaxed), 1);
    EXPECT_EQ(workers[task].load(std::memory_order_relaxed), task % 4);
  }

  pool.parallel_for(7, [&](std::size_t task, std::size_t worker) {
    executions[task].fetch_add(1, std::memory_order_relaxed);
    workers[task].store(worker, std::memory_order_relaxed);
  });
  for (std::size_t task = 0; task < executions.size(); ++task) {
    EXPECT_EQ(executions[task].load(std::memory_order_relaxed),
              task < 7 ? 2 : 1);
    EXPECT_EQ(workers[task].load(std::memory_order_relaxed), task % 4);
  }
} // end - ParallelForExecutesEveryTaskAndCanBeReused

TEST(QecThreadPool, SingleThreadRunsEveryTaskInline) {
  thread_pool pool(/*num_threads=*/1);
  std::array<std::size_t, 5> executions{};
  const std::thread::id caller = std::this_thread::get_id();

  pool.parallel_for(5, [&](std::size_t task, std::size_t worker) {
    ++executions[task];
    EXPECT_EQ(worker, 0);
    EXPECT_EQ(std::this_thread::get_id(), caller);
  });

  EXPECT_EQ(executions, (std::array<std::size_t, 5>{1, 1, 1, 1, 1}));
} // end - SingleThreadRunsEveryTaskInline

TEST(QecThreadPool, ParallelForSkipsEmptyPhase) {
  thread_pool pool(/*num_threads=*/2);
  std::size_t callback_calls = 0;

  pool.parallel_for(0, [&](std::size_t /*task*/, std::size_t /*worker*/) {
    ++callback_calls;
  });
  EXPECT_EQ(callback_calls, 0);

  pool.parallel_for(1, [&](std::size_t task, std::size_t worker) {
    ++callback_calls;
    EXPECT_EQ(task, 0);
    EXPECT_EQ(worker, 0);
  });
  EXPECT_EQ(callback_calls, 1);
} // end - ParallelForSkipsEmptyPhase

TEST(QecThreadPool, ParkedWorkersWakeForALaterPhase) {
  thread_pool pool(/*num_threads=*/4);
  std::atomic<std::size_t> workers{0};

  // Give every owned worker time to enter the between-phase atomic wait, then
  // exercise enough wake/park transitions to catch a missed notification.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  for (std::size_t phase = 0; phase < 100; ++phase) {
    pool.run_workers([&](std::size_t /*worker*/) {
      workers.fetch_add(1, std::memory_order_relaxed);
    });
  }

  EXPECT_EQ(workers.load(std::memory_order_relaxed), 400);
} // end - ParkedWorkersWakeForALaterPhase

TEST(QecThreadPool, WorkerExceptionIsJoinedAndRethrownToCaller) {
  thread_pool pool(/*num_threads=*/4);
  std::atomic<std::size_t> completed{0};
  std::atomic<std::size_t> throwing_worker{4};

  EXPECT_THROW(pool.parallel_for(
                   8,
                   [&](std::size_t task, std::size_t worker) {
                     if (task == 0) {
                       throwing_worker.store(worker, std::memory_order_relaxed);
                       throw std::runtime_error("worker failure");
                     }
                     completed.fetch_add(1, std::memory_order_relaxed);
                   }),
               std::runtime_error);
  EXPECT_EQ(throwing_worker.load(std::memory_order_relaxed), 0);

  completed.store(0, std::memory_order_relaxed);
  EXPECT_NO_THROW(
      pool.parallel_for(8, [&](std::size_t /*task*/, std::size_t /*worker*/) {
        completed.fetch_add(1, std::memory_order_relaxed);
      }));
  EXPECT_EQ(completed.load(std::memory_order_relaxed), 8);
} // end - WorkerExceptionIsJoinedAndRethrownToCaller

TEST(QecThreadPool, RunWorkersPropagatesErrorsAndRemainsReusable) {
  thread_pool pool(/*num_threads=*/4);
  std::atomic<std::size_t> workers{0};

  EXPECT_THROW(pool.run_workers([&](std::size_t worker) {
    if (worker == 1)
      throw std::runtime_error("worker callback failure");
    workers.fetch_add(1, std::memory_order_relaxed);
  }),
               std::runtime_error);

  workers.store(0, std::memory_order_relaxed);
  EXPECT_NO_THROW(pool.run_workers([&](std::size_t /*worker*/) {
    workers.fetch_add(1, std::memory_order_relaxed);
  }));
  EXPECT_EQ(workers.load(std::memory_order_relaxed), 4);
} // end - RunWorkersPropagatesErrorsAndRemainsReusable

TEST(QecThreadPool, ZeroCountAndRoundRobinRespectCpuCap) {
#if !defined(__linux__)
  GTEST_SKIP() << "worker CPU affinity is Linux-only";
#else
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  ASSERT_EQ(pthread_getaffinity_np(pthread_self(), sizeof(allowed), &allowed),
            0);
  int allowed_cpu = -1;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    if (CPU_ISSET(cpu, &allowed)) {
      allowed_cpu = cpu;
      break;
    }
  ASSERT_GE(allowed_cpu, 0);

  thread_pool zero_requested(
      /*num_threads=*/0, {allowed_cpu}, thread_pool::AffinityPolicy::RoundRobin);
  EXPECT_GE(zero_requested.num_threads(), 1u);
  EXPECT_LE(zero_requested.num_threads(), 2u);

  thread_pool capped(/*num_threads=*/8, {allowed_cpu},
                     thread_pool::AffinityPolicy::RoundRobin);
  EXPECT_EQ(capped.num_threads(), 2u);
  std::array<std::atomic<std::size_t>, 4> executions{};
  std::array<std::atomic<std::size_t>, 4> workers{};
  capped.parallel_for(4, [&](std::size_t task, std::size_t worker) {
    executions[task].fetch_add(1, std::memory_order_relaxed);
    workers[task].store(worker, std::memory_order_relaxed);
  });
  for (std::size_t task = 0; task < executions.size(); ++task) {
    EXPECT_EQ(executions[task].load(std::memory_order_relaxed), 1);
    EXPECT_EQ(workers[task].load(std::memory_order_relaxed), task % 2);
  }
#endif
} // end - ZeroCountAndRoundRobinRespectCpuCap

TEST(QecThreadPool, FloatAffinityAppliesCompleteMaskToOwnedWorkers) {
#if !defined(__linux__)
  GTEST_SKIP() << "worker CPU affinity is Linux-only";
#else
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  ASSERT_EQ(pthread_getaffinity_np(pthread_self(), sizeof(allowed), &allowed),
            0);
  int allowed_cpu = -1;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    if (CPU_ISSET(cpu, &allowed)) {
      allowed_cpu = cpu;
      break;
    }
  ASSERT_GE(allowed_cpu, 0);

  thread_pool pool(/*num_threads=*/2, {allowed_cpu},
                   thread_pool::AffinityPolicy::Float);
  cpu_set_t worker_affinity;
  CPU_ZERO(&worker_affinity);
  int affinity_error = -1;
  pool.run_workers([&](std::size_t worker) {
    if (worker == 1)
      affinity_error = pthread_getaffinity_np(
          pthread_self(), sizeof(worker_affinity), &worker_affinity);
  });

  ASSERT_EQ(affinity_error, 0);
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
    EXPECT_EQ(CPU_ISSET(cpu, &worker_affinity) != 0, cpu == allowed_cpu);
#endif
} // end - FloatAffinityAppliesCompleteMaskToOwnedWorkers

TEST(QecThreadPool, StartupAffinityFailureJoinsWorkers) {
#if !defined(__linux__)
  GTEST_SKIP() << "worker CPU affinity is Linux-only";
#else
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  ASSERT_EQ(pthread_getaffinity_np(pthread_self(), sizeof(allowed), &allowed),
            0);
  int allowed_cpu = -1;
  int unavailable_cpu = -1;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &allowed)) {
      if (allowed_cpu < 0)
        allowed_cpu = cpu;
    } else if (unavailable_cpu < 0) {
      unavailable_cpu = cpu;
    }
  }
  ASSERT_GE(allowed_cpu, 0);
  if (unavailable_cpu < 0)
    GTEST_SKIP() << "all representable CPUs are available to this process";

  EXPECT_THROW((void)thread_pool(
                   /*num_threads=*/3, {allowed_cpu, unavailable_cpu},
                   thread_pool::AffinityPolicy::RoundRobin),
               std::runtime_error);
#endif
} // end - StartupAffinityFailureJoinsWorkers

} // namespace
} // namespace cudaq::qec::detail
