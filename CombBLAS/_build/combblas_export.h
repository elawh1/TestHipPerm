
#ifndef COMBBLAS_EXPORT_H
#define COMBBLAS_EXPORT_H

#ifdef COMBBLAS_STATIC_DEFINE
#  define COMBBLAS_EXPORT
#  define COMBBLAS_NO_EXPORT
#else
#  ifndef COMBBLAS_EXPORT
#    ifdef CombBLAS_EXPORTS
        /* We are building this library */
#      define COMBBLAS_EXPORT 
#    else
        /* We are using this library */
#      define COMBBLAS_EXPORT 
#    endif
#  endif

#  ifndef COMBBLAS_NO_EXPORT
#    define COMBBLAS_NO_EXPORT 
#  endif
#endif

#ifndef COMBBLAS_DEPRECATED
#  define COMBBLAS_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef COMBBLAS_DEPRECATED_EXPORT
#  define COMBBLAS_DEPRECATED_EXPORT COMBBLAS_EXPORT COMBBLAS_DEPRECATED
#endif

#ifndef COMBBLAS_DEPRECATED_NO_EXPORT
#  define COMBBLAS_DEPRECATED_NO_EXPORT COMBBLAS_NO_EXPORT COMBBLAS_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef COMBBLAS_NO_DEPRECATED
#    define COMBBLAS_NO_DEPRECATED
#  endif
#endif

#endif /* COMBBLAS_EXPORT_H */
