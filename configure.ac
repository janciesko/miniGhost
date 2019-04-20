AC_INIT([minighost], [.01], [nevans@sandia.gov])
	AM_INIT_AUTOMAKE([-Wall foreign]) # -Werror
	AC_PROG_CC
	AM_PROG_CC_C_O
	AC_PROG_FC

AC_ARG_WITH([mpi],
		[AS_HELP_STRING([--with-mpi],
			[choose which mpi implementation to use (mpi, mpiq)])])

AS_IF([test "x$with_mpi" != "x"],
	[case "$with_mpi" in
	mpi)
		AC_DEFINE([_MG_MPI], [1], [Use conventional MPI])
		;;
	mpiq)
		AC_DEFINE([_MG_MPI], [1], [Defined to use MPI with MPIQ])
		AC_DEFINE([_MG_MPIQ], [1], [Defined to use experimental MPIQ macros from qthreads])
		;;
	*)
		AC_MSG_ERROR([Unknown MPI type])
		;;
	esac])

AC_ARG_ENABLE([debug],
		[AS_HELP_STRING([--enable-debug],
			[Enable Minighost debug options])])

AS_IF([test "x$enable_debug" = "xyes"],
      [AC_DEFINE([_MG_DEBUG], [1], [Defined if debug support is enabled])])

AC_ARG_ENABLE([qthreads-profiling],
		[AS_HELP_STRING([--enable-qthreads-profiling],
			[Enable Qthreads profiling])])

AS_IF([test "x$enable_qthreads_profiling" = "xyes"],
      [AC_DEFINE([_MG_QT_PROFILE], [1], [Defined if Qthreads profiling support is enabled])])

AC_ARG_WITH([timing],
    	    [AS_HELP_STRING([--with-timing],
			[choose which timing method to use (l0, l1)])])

AS_IF([test "x$with_timing" != "x"],
      [case "$with_timing" in
			l0)
AC_DEFINE([_MG_TIMING_L0], [1], [Defined to measure high level timings])
	;;
	l1)
AC_DEFINE([_MG_TIMING_L1], [1], [Defined to measure low level timings])
	;;
	*)
AC_MSG_ERROR([Unknown timing type])
	;;
	esac])

AC_ARG_ENABLE([craypat-api],
		[AS_HELP_STRING([--enable-craypat-api],
			[Enable craypat api profiling])])

	AS_IF([test "x$enable_qthreads_profiling" = "xyes"],
			[AC_DEFINE([_USE_PAT_API], [1], [Defined if CrayPat API support is enabled])])

AC_ARG_WITH([runtime],
		[AS_HELP_STRING([--with-runtime],
			[choose which runtime to use (serial, openmp, qthreads, argobots)])])

	AS_IF([test "x$with_runtime" != "x"],
			[case "$with_runtime" in
			serial)
AC_DEFINE([_MG_SERIAL], [1], [Defined to execute serially])
	;;
	openmp)
AC_DEFINE([_MG_OPENMP], [1], [Defined to use OpenMP runtime])
	;;
	qthreads)
	AC_DEFINE([_MG_QT], [1], [Defined to use Qthreads runtime])
	AC_CHECK_HEADER([qthread/qthread.h],[],AC_MSG_ERROR([unable to find Qthreads header file]))
	AC_CHECK_LIB([qthread],[qthread_fork],[],AC_MSG_ERROR([unable to find Qthreads library]))
	;;
	argobots)
AC_DEFINE([_MG_ARGOBOTS], [1], [Defined to use Argobots runtime])
	AC_CHECK_HEADER([abt.h],[],AC_MSG_ERROR([unable to find Argobots header file]))
	AC_CHECK_LIB([abt],[ABT_init],[],AC_MSG_ERROR([unable to find Argobots library]))
	;;

	*)
AC_MSG_ERROR([Unknown runtime])
	;;
	esac])

AC_CONFIG_HEADERS([config.h])
	AC_CONFIG_FILES([
			Makefile
			src/Makefile
			])
			#ref/Makefile
	AC_OUTPUT
