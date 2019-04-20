// ************************************************************************
//
//          miniGhost: stencil computations with boundary exchange.
//                 Copyright (2013) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Richard F. Barrett (rfbarre@sandia.gov) or
//                    Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************

#define MG_EXTERN

#include "mg_tp.h"

#if defined _MG_QT_PROFILE
aligned_t num_tasks;
#endif

#if defined _MG_QT
typedef struct {
   int           ivar;
   StateVar   ** g;
   BlockInfo   * blk_iblk;
   InputParams * params;
} compute_block_args_t;

aligned_t compute_block(void * args_)
{
   compute_block_args_t * args = (compute_block_args_t *)args_;

   int            const ivar     = args->ivar;
   StateVar    ** const g        = args->g;
   BlockInfo    * const blk_iblk = args->blk_iblk;
   InputParams  * const params   = args->params;

   int ierr = 0;

   MG_Stencil ( *params, g, *blk_iblk, ivar );

#    if defined _MG_QT_PROFILE
   qthread_incr(&num_tasks, 1);
#    endif

   return ierr;
}

#elif defined _MG_ARGOBOTS

void tp_abt_init ( int num_vars, int num_blks );
void tp_abt_finalize ();
void tp_abt_kernel ( StateVar **g, BlockInfo **blk, InputParams *params,
                     int irange, int jrange );

#endif

int main ( int argc, char* argv[] )
{
   int
     count,                        // Array length.
     i,
     iblk, ivar,                   // Counters
     itask,                        // Counter
     tstep,                        // Counter
     ierr,                         // Return value
     ispike,                       // Counter
     ithread,                      // Counter
     len,                          // Temporary var for computing thread block offsets.
     mype,
     nblks_xdir,                   // Number of blocks in x direction.
     nblks_ydir,                   // Number of blocks in y direction.
     nblks_zdir,                   // Number of blocks in z direction.
     numthreads,
     num_errors = 0,               // Number of variables in error.
     rem_x,                        // Remainder of domain in x direction.
     rem_y,                        // Remainder of domain in y direction.
     rem_z,                        // Remainder of domain in z direction.
     rootpe,
     tasks_per_tstep;              // Number of tasks to be spawned per time step.

   double 
      time_start;

   InputParams 
      params;             // Problem parameters.

   // ---------------------
   // Executable Statements
   // ---------------------

   memory_stats.numallocs = 0; // Struct instantiated in mg_perf.h
   memory_stats.count     = 0;
   memory_stats.bytes     = 0;

#if defined (_MG_QT) && !defined (_MG_MPIQ)
            qthread_initialize();
#elif defined (_MG_ARGOBOTS)
            ABT_init(0, 0);
#endif

   ierr = MG_Init ( argc, argv, &params );
   MG_Assert ( !ierr, "main:MG_Init" );

   mype       = mgpp.mype;
   numthreads = mgpp.num_threads;
   rootpe     = mgpp.rootpe;

   tasks_per_tstep = params.numvars*params.numblks;

   // Print problem information to stdout.
   ierr = MG_Print_header ( params );

   // Allocate and configure sub-blocks.

   params.numblks = ( params.nx / params.blkxlen ) *
                    ( params.ny / params.blkylen ) *
                    ( params.nz / params.blkzlen );

   blk = (BlockInfo**)MG_CALLOC ( params.numblks, sizeof(BlockInfo*) );
   MG_Assert ( !ierr, "main: Allocation of **blk" );

#if defined (_MG_ARGOBOTS)
   tp_abt_init ( params.numvars, params.numblks );
#endif

   ierr = MG_Block_init ( &params, blk );
   MG_Assert ( !ierr, "main: MG_Block_init" );

#if defined _MG_OPENMP

   params.thread_offset_xy = (int*)MG_CALLOC ( mgpp.num_threads, sizeof(int) );
   MG_Assert ( !ierr, "main: Allocation of params.thread_offset_xy" );

   params.thread_offset_xz = (int*)MG_CALLOC ( mgpp.num_threads, sizeof(int) );
   MG_Assert ( !ierr, "main: Allocation of params.thread_offset_xz" );

   params.thread_offset_yz = (int*)MG_CALLOC ( mgpp.num_threads, sizeof(int) );
   MG_Assert ( !ierr, "main: Allocation of params.thread_offset_yz" );

   for ( i=0; i<mgpp.num_threads; i++ ) {
      len = params.blkxlen * ( params.blkylen / mgpp.num_threads ); // For use in boundary exchange routines.
      params.thread_offset_xy[i] = len * i;

      len = params.blkxlen * ( params.blkzlen / mgpp.num_threads ); // For use in boundary exchange routines.
      params.thread_offset_xz[i] = len * i;

      len = params.blkylen * ( params.blkzlen / mgpp.num_threads ); // For use in boundary exchange routines.
      params.thread_offset_yz[i] = len * i;

      //printf ( "[pe %d:%d] offsets = (%d, %d, %d) \n", mgpp.mype, mgpp.thread_id, params.thread_offset_xy[i], params.thread_offset_xz[i], params.thread_offset_yz[i] );
   }
   MG_Barrier ();

#endif

   // Allocate and initialize state variables.
   g = (StateVar**)MG_CALLOC ( params.numvars, sizeof(StateVar*) );
   if ( g == NULL ) {
      fprintf ( stderr, "Allocation of arrays of g %d failed \n",
                params.numvars*(int)sizeof(StateVar*) );
      ierr =  -1;
      MG_Assert ( !ierr, "main: Allocation of **g" );
   }

   ierr = MG_Grid_init ( &params, g );
   MG_Assert ( !ierr, "main: MG_Grid_init" );

   // Allocate and set up spikes.
   spikes = (SpikeInfo**)MG_CALLOC ( params.numspikes, sizeof(SpikeInfo*) );
   if ( spikes == NULL ) {
      fprintf ( stderr, "Allocation of spikes of size %d failed \n", params.numspikes );
      MG_Assert ( spikes != NULL, "main: Allocation of array of structs spikes" );
   }
   ierr = MG_Spike_init ( params, spikes );
   MG_Assert ( !ierr, "main: MG_Spike_init" );

   tstep = 1; // Allows for some initialization, i.e. spike insertion.

   // Synchronize for timing.
   ierr = MG_Barrier ( );
   MG_Assert ( !ierr, "main:MG_Barrier" );

#if defined _USE_PAT_API
   ierr = PAT_region_begin ( 1, "miniGhost" );
   MG_Assert ( !ierr, "main: PAT_region_begin" );
#endif

#if defined HAVE_GEMINI_COUNTERS
   ierr = gpcd_start ( );
   MG_Assert ( !ierr, "main: gpcd_start" );
#endif

   MG_Time_start(time_start);

   for ( ispike=0; ispike<params.numspikes; ispike++ ) {

      ierr = MG_Spike_insert ( params, spikes, ispike, g, tstep );
      MG_Assert ( !ierr, "main:MG_Spike_insert" );

      itask = 0;
      for ( tstep=1; tstep<=params.numtsteps; tstep++ ) { // Time step loop.

#if defined _MG_QT
            compute_block_args_t compute_block_args;
            qt_sinc_t sinc;
            qt_sinc_init(&sinc, 0, NULL, NULL, params.numvars * params.numblks);
#endif

#if !defined _MG_ARGOBOTS
         for ( ivar=0; ivar<params.numvars; ivar++ ) {   // Loop over variables.

            for ( ithread=0; ithread<numthreads; ithread++ ) {
               g[ivar]->thr_flux[ithread] = 0.0;
            }
            //MG_Barrier ( );

            // Spawn tasks, one per domain subblk.
            for ( iblk=0; iblk<params.numblks; iblk++ ) {

#if defined _MG_SERIAL || defined _MG_MPI
#  if defined _MG_QT // {
                //compute_block_args_t const compute_block_args = { ivar, iblk, g[ivar], blk[iblk], params};
                compute_block_args.ivar     = ivar;
                compute_block_args.g        = g;
                compute_block_args.blk_iblk = blk[iblk];
                compute_block_args.params   = &params;

                {
                    unsigned int          task_flags;
                    qthread_shepherd_id_t target_shep;

                    task_flags = QTHREAD_SPAWN_RET_SINC_VOID;
                    if (1 == blk[iblk]->info) {
                        task_flags |= QTHREAD_SPAWN_SIMPLE;
                        target_shep = NO_SHEPHERD; 
                    } else {
                        target_shep = 0;
                    }
                    qthread_spawn ( compute_block, 
                            &compute_block_args, 
                            sizeof(compute_block_args_t), 
                            &sinc,
                            0,    /* no preconds */
                            NULL, /* no preconds */
                            target_shep,
                            task_flags);
                }
#  else        // MPI everywhere and MPI + OpenMP. } {
               ierr = MG_Stencil ( params, g, *blk[iblk], ivar );
               MG_Assert ( !ierr, "main:MG_Stencil" );
#  endif // }
#elif defined _MG_QT
#endif
            } // End loop over blks.

         } // end params.numvars

#if defined _MG_QT
         qt_sinc_wait ( &sinc, NULL );
         qt_sinc_fini ( &sinc );
#endif

#else // _MG_ARGOBOTS

         for ( ivar=0; ivar<params.numvars; ivar++ ) {
            for ( ithread=0; ithread<numthreads; ithread++ ) {
               g[ivar]->thr_flux[ithread] = 0.0;
            }
         }
         tp_abt_kernel ( g, blk, &params, params.numvars, params.numblks );

#endif

         /* Toggle variable domains _after_ synchronizing all tasks in
          * this time step. */
         for ( ivar=0; ivar<params.numvars; ivar++ ) {   // Loop over variables.
            g[ivar]->toggle++;
            g[ivar]->toggle = g[ivar]->toggle % 2;
         }

         // Correctness check across all variables.

         num_errors = MG_Sum_grid ( params, g, tstep, 0 );
         if ( num_errors ) {
            if ( mype == rootpe ) {
               fprintf ( stderr, 
                         "\n *** Iteration not correct; %d errors reported after iteration %d.\n\n",
                         num_errors, tstep );
            }
            MG_Assert ( !num_errors, "main: Terminating execution" );
         }
         else {
            if ( mype == rootpe ) {
               if ( tstep % params.check_answer_freq == 0 ) {
                  fprintf ( stdout, " End time step %d for spike %d. \n", tstep, ispike+1 );
               }
            }
         }
         // FIXME: Inserted for testing.
         ierr = MG_Barrier ();
      } // End tsteps

   } // End spike insertion.

#if defined _MG_ARGOBOTS
   tp_abt_finalize ();
#endif

#if defined _USE_PAT_API
   ierr = PAT_region_end ( 1 );
   MG_Assert ( !ierr, "main: PAT_region_end" );
#endif

   MG_Time_accum(time_start,timings.total);

#if defined HAVE_GEMINI_COUNTERS
   ierr = gpcd_end ( );
   MG_Assert ( !ierr, "main: gpcd_end" );
#endif

   // Final correctness check.
   num_errors = MG_Sum_grid ( params, g, tstep+1, 1 );
   if ( mype == rootpe ) {
      if ( !num_errors )  {
         fprintf ( stdout, "\n\n ** Final correctness check PASSED. ** \n\n" );
      }
      else {
         fprintf ( stdout, "\n\n ** Final correctness check FAILED. (%d variables showed errors) ** \n\n", 
                   num_errors );
      }
   }
   ierr = MG_Report_performance ( params );
   MG_Assert ( !ierr, "main: MG_Report_performance" );

   // Release workspace.

   if ( g != NULL )  {
      free ( g );
   }
   if ( blk != NULL )  {
      free ( blk );
   }

   ierr = MG_Terminate ( );
   MG_Assert ( !ierr, "main: MG_Terminate" );

   exit(0);

// End main.c
}

#if defined _MG_ARGOBOTS

// POWER uses 128 while the others 64
#ifdef __powerpc__
#  define MG_TP_ABT_CACHELINE_SIZE 128
#else
#  define MG_TP_ABT_CACHELINE_SIZE 64
#endif

typedef struct {
   ABT_thread    handle;
   int           ifrom, ito, jfrom, jto;
   StateVar   ** g;
   BlockInfo  ** blk;
   InputParams * params;
   char padding[MG_TP_ABT_CACHELINE_SIZE-sizeof(void *)*4-sizeof(int)*4];
} tp_abt_thread_data_t;

struct {
   char padding1[MG_TP_ABT_CACHELINE_SIZE];
   int num_xstreams;
   int num_threads;
   int num_vars, num_blks;
   int split_a, split_b;
   ABT_xstream *xstreams;
   ABT_sched *scheds;
   ABT_pool *priv_pools;
   ABT_pool *shared_pools;
   tp_abt_thread_data_t *thread_data;
   char padding2[MG_TP_ABT_CACHELINE_SIZE];
} g_abt_global_data;

static int tp_abt_sched_init ( ABT_sched sched, ABT_sched_config config );
static void tp_abt_sched_run ( ABT_sched sched );
static int tp_abt_sched_free ( ABT_sched sched );

static inline void *tp_abt_malign ( size_t size ) {
    void *p_ptr;
    int ret = posix_memalign ( &p_ptr, MG_TP_ABT_CACHELINE_SIZE, size );
    assert ( ret == 0 );
    return p_ptr;
}

static inline void tp_abt_size_check () {
   // size check
   int vals[1 - (sizeof(tp_abt_thread_data_t) % MG_TP_ABT_CACHELINE_SIZE == 0
                 ? 0 : 2)];
   (void) vals;
}

void tp_abt_init ( int num_vars, int num_blks ) {
   int i, j, rank;
   char *env = getenv ( "ABT_NUM_XSTREAMS" );
   int num_xstreams = 0;
   if ( env ) {
      num_xstreams = atoi ( env );
   } else {
      num_xstreams = sysconf ( _SC_NPROCESSORS_ONLN );
   }
   printf("num_vars %d num_blks = %d\n", num_vars, num_blks);
   g_abt_global_data.num_xstreams = num_xstreams;
   g_abt_global_data.num_vars = num_vars;
   g_abt_global_data.num_blks = num_blks;

   // Create thread pools
   g_abt_global_data.priv_pools = (ABT_pool *)
      tp_abt_malign ( sizeof(ABT_pool) * num_xstreams );
   g_abt_global_data.shared_pools = (ABT_pool *)
      tp_abt_malign ( sizeof(ABT_pool) * num_xstreams );
   for ( i = 0; i < num_xstreams; i++) {
      ABT_pool_create_basic ( ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC, ABT_TRUE,
                              &g_abt_global_data.priv_pools[i] );
   }
   for ( i = 0; i < num_xstreams; i++) {
      ABT_pool_create_basic ( ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC, ABT_TRUE,
                              &g_abt_global_data.shared_pools[i] );
   }

   // Create schedulers
   g_abt_global_data.scheds = (ABT_sched *)
      tp_abt_malign ( sizeof(ABT_sched) * num_xstreams );
   {
      ABT_sched_def sched_def;
      sched_def.type = ABT_SCHED_TYPE_ULT;
      sched_def.init = tp_abt_sched_init;
      sched_def.run = tp_abt_sched_run;
      sched_def.free = tp_abt_sched_free;
      sched_def.get_migr_pool = NULL;

      ABT_sched_config_var cv_freq = {
         .idx = 0,
         .type = ABT_SCHED_CONFIG_INT
      };
      ABT_sched_config config;
      ABT_sched_config_create ( &config, cv_freq, 10000,
                                ABT_sched_config_var_end );

      ABT_pool *my_pools = (ABT_pool *)
         malloc ( sizeof(ABT_pool) * (num_xstreams + 1) );
      for ( rank = 0; rank < num_xstreams; rank++ ) {
         my_pools[0] = g_abt_global_data.priv_pools[rank];
         for ( i = 0; i < num_xstreams; i++ ) {
            my_pools[i + 1]
               = g_abt_global_data.shared_pools[(rank + i) % num_xstreams];
         }
         ABT_sched_create ( &sched_def, num_xstreams + 1, my_pools, config,
                            &g_abt_global_data.scheds[rank] );
      }
      free ( my_pools );
      ABT_sched_config_free ( &config );
   }

   // Create execution streams (=workers) and thread pools
   g_abt_global_data.xstreams = (ABT_xstream *)
      tp_abt_malign ( sizeof(ABT_xstream) * num_xstreams );
   for ( rank = 0; rank < num_xstreams; rank++ ) {
      ABT_xstream xstream;
      if ( rank == 0 ) {
         ABT_xstream_self ( &g_abt_global_data.xstreams[rank] );
         ABT_xstream_set_main_sched ( g_abt_global_data.xstreams[rank],
                                      g_abt_global_data.scheds[rank] );
      } else {
         ABT_xstream_create ( g_abt_global_data.scheds[rank],
                              &g_abt_global_data.xstreams[rank] );
      }
   }

   // Create threads.
   g_abt_global_data.thread_data = (tp_abt_thread_data_t *)
      tp_abt_malign ( sizeof(tp_abt_thread_data_t) * num_vars * num_blks );
   for ( i = 0; i < num_vars * num_blks; i++ ) {
      g_abt_global_data.thread_data[i].handle = ABT_THREAD_NULL;
   }

   // Find good values to decompose the range.
   for ( i = 1; i < num_xstreams + 1; i++ ) {
      int ii = i * i;
      if ( ( ii - ( i << 1 ) + 1 ) < num_xstreams && ii >= num_xstreams )
         break;
   }

   for ( j = (i > num_vars ? num_vars : i); j >= 0; j-- ) {
      if ( num_xstreams % j == 0 ) {
         if (j > num_blks) {
           g_abt_global_data.split_a = num_xstreams / num_blks;
           g_abt_global_data.split_b = num_blks;
         } else {
           g_abt_global_data.split_a = num_xstreams / j;
           g_abt_global_data.split_b = j;
         }
         break;
      }
   }
   g_abt_global_data.num_threads = num_vars * num_blks;

   // Adjust the number of threads.
   mgpp.num_threads = mg_get_num_os_threads();
}

void tp_abt_finalize () {
   int i;
   // Free threads.
   for ( i = 0; i < g_abt_global_data.num_threads;
         i++ ) {
      if ( g_abt_global_data.thread_data[i].handle != ABT_THREAD_NULL ) {
         ABT_thread_free ( &g_abt_global_data.thread_data[i].handle );
      }
   }
   // Free secondary execution streams
   for ( i = 1; i < g_abt_global_data.num_xstreams; i++ ) {
      ABT_xstream_join ( g_abt_global_data.xstreams[i] );
      ABT_xstream_free ( &g_abt_global_data.xstreams[i] );
   }
   // Free schedulers of secondary execution streams
   for ( i = 1; i < g_abt_global_data.num_xstreams; i++ ) {
      ABT_sched_free ( &g_abt_global_data.scheds[i] );
   }
   free ( g_abt_global_data.thread_data );
   free ( g_abt_global_data.scheds );
   free ( g_abt_global_data.priv_pools );
   free ( g_abt_global_data.shared_pools );
   free ( g_abt_global_data.xstreams );
}

void tp_abt_create_thread ( int ifrom, int ito, int jfrom, int jto, int depth,
                            int index, StateVar **g, BlockInfo **blk,
                            InputParams *params );
void tp_abt_join_thread ( int ifrom, int jfrom );

int mg_get_num_os_threads () {
  if ( g_abt_global_data.num_xstreams == 0 ) {
    char *env = getenv ( "ABT_NUM_XSTREAMS" );
    int num_xstreams = 0;
    if ( env ) {
       num_xstreams = atoi ( env );
    } else {
       num_xstreams = sysconf ( _SC_NPROCESSORS_ONLN );
    }
    g_abt_global_data.num_xstreams = num_xstreams;
    return num_xstreams;
  } else {
    return g_abt_global_data.num_xstreams;
  }
}

int mg_get_os_thread_num () {
   int rank = 0;
   ABT_xstream_self_rank(&rank);
   return rank;
}

void compute_block ( void * args_ )
{
   tp_abt_thread_data_t * args = (tp_abt_thread_data_t *)args_;
   int ii, jj;

   const int ifrom = args->ifrom;
   const int ito = args->ito;
   const int jfrom = args->jfrom;
   const int jto = args->jto;

   StateVar **g = args->g;
   BlockInfo **blk = args->blk;
   InputParams *params = args->params;

   int irange = ito - ifrom;
   int jrange = jto - jfrom;
   if ( irange == 0 || jrange == 0 )
      return;

   if ( irange == 1 ) {
      if ( jrange == 1 ) {
         // Leaf node.
         MG_Stencil ( *params, g, *blk[jfrom], ifrom );
      } else {
         // Iterate over J (2-way)
         const int jfrom1 = jfrom;
         const int jto1 = jfrom + (jrange >> 1);
         const int jfrom2 = jto1;
         const int jto2 = jto;
         tp_abt_create_thread ( ifrom, ito, jfrom2, jto2,
                                1, 0, g, blk, params );
         tp_abt_create_thread ( ifrom, ito, jfrom1, jto1,
                                1, -1, g, blk, params );
         tp_abt_join_thread ( ifrom, jfrom2 );
      }
   } else {
      if ( jrange == 1 ) {
         // Iterate over I (2-way)
         const int ifrom1 = ifrom;
         const int ito1 = ifrom + (irange >> 1);
         const int ifrom2 = ito1;
         const int ito2 = ito;
         tp_abt_create_thread ( ifrom2, ito2, jfrom, jto,
                                1, 0, g, blk, params );
         tp_abt_create_thread ( ifrom1, ito1, jfrom, jto,
                                1, -1, g, blk, params );
         tp_abt_join_thread ( ifrom2, jfrom );
      } else {
         // Iterate over I and J (2,2-way)
         const int ifrom1 = ifrom;
         const int ito1 = ifrom + (irange >> 1);
         const int ifrom2 = ito1;
         const int ito2 = ito;
         const int jfrom1 = jfrom;
         const int jto1 = jfrom + (jrange >> 1);
         const int jfrom2 = jto1;
         const int jto2 = jto;
         tp_abt_create_thread ( ifrom2, ito2, jfrom2, jto2,
                                1, 0, g, blk, params );
         tp_abt_create_thread ( ifrom1, ito1, jfrom2, jto2,
                                1, 0, g, blk, params );
         tp_abt_create_thread ( ifrom2, ito2, jfrom1, jto1,
                                1, 0, g, blk, params );
         tp_abt_create_thread ( ifrom1, ito1, jfrom1, jto1,
                                1, -1, g, blk, params );

         tp_abt_join_thread ( ifrom2, jfrom2 );
         tp_abt_join_thread ( ifrom2, jfrom1 );
         tp_abt_join_thread ( ifrom1, jfrom2 );
      }
   }
}

void tp_abt_create_thread ( int ifrom, int ito, int jfrom, int jto, int depth,
                            int index, StateVar **g, BlockInfo **blk,
                            InputParams *params ) {
   ABT_pool target_pool = ABT_POOL_NULL;
   if ( depth == 0 ) {
      // Assign to the private pool.
      target_pool = g_abt_global_data.priv_pools[index];
   } else if ( index >= 0 ) {
      // Assign to the local shared pool.
      int rank;
      ABT_xstream_self_rank ( &rank );
      target_pool = g_abt_global_data.shared_pools[rank];
   }
   int data_idx = ifrom * g_abt_global_data.num_blks + jfrom;

   // Set arguments.
   g_abt_global_data.thread_data[data_idx].ifrom = ifrom;
   g_abt_global_data.thread_data[data_idx].ito = ito;
   g_abt_global_data.thread_data[data_idx].jfrom = jfrom;
   g_abt_global_data.thread_data[data_idx].jto = jto;
   g_abt_global_data.thread_data[data_idx].g = g;
   g_abt_global_data.thread_data[data_idx].blk = blk;
   g_abt_global_data.thread_data[data_idx].params = params;

   // Create threads.
   if ( target_pool != ABT_POOL_NULL ) {
      if ( g_abt_global_data.thread_data[data_idx].handle != ABT_THREAD_NULL ) {
         ABT_thread_revive ( target_pool, compute_block,
                             &g_abt_global_data.thread_data[data_idx],
                             &g_abt_global_data.thread_data[data_idx].handle);
      } else {
         ABT_thread_create ( target_pool, compute_block,
                             &g_abt_global_data.thread_data[data_idx],
                             ABT_THREAD_ATTR_NULL,
                             &g_abt_global_data.thread_data[data_idx].handle);
      }
   } else {
      // Serialize it.
      compute_block ( &g_abt_global_data.thread_data[data_idx] );
   }
}

void tp_abt_join_thread ( int ifrom, int jfrom ) {
   int data_idx = ifrom * g_abt_global_data.num_blks + jfrom;
   ABT_thread_join ( g_abt_global_data.thread_data[data_idx].handle );
}

void tp_abt_kernel ( StateVar **g, BlockInfo **blk, InputParams *params,
                     int irange, int jrange ) {
   int ii, jj;
   const int split_a = g_abt_global_data.split_a;
   const int split_b = g_abt_global_data.split_b;

   int prev_ito = 0;
   for ( ii = 0; ii < split_a; ii++ ) {
      int ifrom = prev_ito;
      int ito = (irange * (ii + 1)) / split_a;
      int prev_jto = 0;
      for ( jj = 0; jj < split_b; jj++ ) {
         int jfrom = prev_jto;
         int jto = (jrange * (jj + 1)) / split_b;
         prev_jto = jto;
         if (ito - ifrom > 0 && jto - jfrom > 0) {
            tp_abt_create_thread ( ifrom, ito, jfrom, jto, 0, (ii + jj * split_a),
                                   g, blk, params );
         }
      }
      prev_ito = ito;
   }

   prev_ito = 0;
   for ( ii = 0; ii < split_a; ii++ ) {
      int ifrom = prev_ito;
      int ito = (irange * (ii + 1)) / split_a;
      int prev_jto = 0;
      for ( jj = 0; jj < split_b; jj++ ) {
         int jfrom = prev_jto;
         int jto = (jrange * (jj + 1)) / split_b;
         prev_jto = jto;
         if (ito - ifrom > 0 && jto - jfrom > 0) {
            tp_abt_join_thread ( ifrom, jfrom );
         }
      }
      prev_ito = ito;
   }
}

static inline uint32_t tp_abt_fast_rand ( uint32_t *p_seed ) {
  // Xorshift
  uint32_t seed = *p_seed;
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  *p_seed = seed;
  return seed;
}

static int tp_abt_sched_init ( ABT_sched sched, ABT_sched_config config ) {
   return ABT_SUCCESS;
}

static void tp_abt_sched_run ( ABT_sched sched ) {
   int rank;
   ABT_xstream_self_rank ( &rank );

   uint32_t seed = 0;
   while ( seed == 0 ) { // seed may not be 0.
     seed = time ( NULL ) + rank * 777;
   }

   int num_pools;
   ABT_sched_get_num_pools ( sched, &num_pools );
   ABT_pool *pools = (ABT_pool *)alloca ( num_pools * sizeof(ABT_pool) );
   ABT_sched_get_pools ( sched, num_pools, 0, pools );

   ABT_pool priv_pool = pools[0];
   ABT_pool my_shared_pool = pools[1];
   int num_shared_pools = num_pools - 2;
   ABT_pool* shared_pools = &pools[2];

   int run_cnt = 0, work_count = 0;
   while ( 1 ) {
      int local_run_cnt = 0;
      ABT_unit unit = ABT_UNIT_NULL;
      // Private pool.
      ABT_pool_pop ( priv_pool, &unit );
      if ( unit != ABT_UNIT_NULL ) {
         ABT_xstream_run_unit ( unit, priv_pool );
         run_cnt++; local_run_cnt++;
      }
      // My shared pool.
      ABT_pool_pop ( my_shared_pool, &unit );
      if ( unit != ABT_UNIT_NULL ) {
         ABT_xstream_run_unit ( unit, my_shared_pool );
         run_cnt++; local_run_cnt++;
      }
      if ( num_shared_pools >= 1 &&
           (local_run_cnt == 0 || (work_count & 1023) == 0) ) {
         // Remote pools.
         uint32_t rand_val = tp_abt_fast_rand ( &seed );
         int target = rand_val % num_shared_pools;
         ABT_pool_pop ( shared_pools[target], &unit );
         if ( unit != ABT_UNIT_NULL ) {
            ABT_unit_set_associated_pool ( unit, my_shared_pool );
            ABT_xstream_run_unit ( unit, my_shared_pool );
         }
      }
      if ( work_count++ >= 10000 ) {
         ABT_xstream_check_events ( sched );
         ABT_bool stop;
         ABT_sched_has_to_stop ( sched, &stop );
         if ( stop == ABT_TRUE ) {
            break;
         } else {
            if ( run_cnt < 50 ) {
               // sched_yield ();
            }
            run_cnt = 0;
         }
      }
   }
}

static int tp_abt_sched_free ( ABT_sched sched ) {
   return ABT_SUCCESS;
}

#endif // _MG_ARGOBOTS
