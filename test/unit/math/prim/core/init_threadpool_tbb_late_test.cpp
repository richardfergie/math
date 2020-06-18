#include <stan/math/prim/core.hpp>

#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <test/unit/math/prim/functor/utils_threads.hpp>

#include <tbb/task_scheduler_init.h>
#include <tbb/task_arena.h>

TEST(intel_tbb_late_init, check_status) {
  const int num_threads = tbb::task_scheduler_init::default_num_threads();
  tbb::task_scheduler_init tbb_scheduler;

  if (num_threads > 1) {
    tbb::task_scheduler_init& tbb_init = stan::math::init_threadpool_tbb(num_threads - 1);
    EXPECT_TRUE(tbb_init.is_active());

    // the max_threads argument for init_threadpool_tbb is not being
    // honored if we have already initialized the TBB scheduler
    EXPECT_EQ(num_threads, tbb::this_task_arena::max_concurrency());
    tbb_init.terminate();
    EXPECT_FALSE(tbb_init.is_active());
  }  
}

// tests with deprecated STAN_NUM_THREADS env. variable
TEST(intel_tbb_late_init, check_status_deprecated) {
  const int num_threads = tbb::task_scheduler_init::default_num_threads();
  tbb::task_scheduler_init tbb_scheduler;

  if (num_threads > 1) {
    set_n_threads(num_threads - 1);
    tbb::task_scheduler_init& tbb_init = stan::math::init_threadpool_tbb();
    EXPECT_TRUE(tbb_init.is_active());

    // STAN_NUM_THREADS is not being honored if we have first
    // initialized the TBB scheduler outside of init_threadpool_tbb
    EXPECT_EQ(num_threads, tbb::this_task_arena::max_concurrency());
  }
}
