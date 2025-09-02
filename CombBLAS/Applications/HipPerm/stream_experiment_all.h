#include <mpi.h>
#include <stdint.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <cmath>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/SpParHelper.h"
#include "CombBLAS/MultiwayMerge.h"
#include <inttypes.h>

#include <unordered_set>

using namespace std;
using namespace combblas;

class Dist
{
public:
    typedef SpDCCols<int64_t, double> DCCols;
    typedef SpParMat<int64_t, double, DCCols> MPI_DCCols;
};



int ExpandPermute(Dist::MPI_DCCols &M11, FullyDistVec<int, int64_t> &perm_pvec_M11, FullyDistVec<int, int64_t> &perm_qvec_M11, int64_t maxnrow, int64_t maxncol, MPI_Comm World) {
    
    int nthreads = 1;
#ifdef THREADED
#pragma omp parallel
{
    nthreads = omp_get_num_threads();
}
#endif
    // Define types for combBLAS
    typedef PlusTimesSRing<double, double> PTFF;

    // double rsstrt, rsend; 
    // MPI_Barrier(World);
    // rsstrt = MPI_Wtime();
    
    int nprocs, myrank;
    MPI_Comm_size(World,&nprocs);
    MPI_Comm_rank(World, &myrank);
    auto M11commGrid = M11.getcommgrid();
    int M11gridrows = M11commGrid->GetGridRows();
    int M11gridcols = M11commGrid->GetGridCols();
    MPI_Comm ColWorld = M11commGrid->GetColWorld();
    MPI_Comm RowWorld = M11commGrid->GetRowWorld();

    int rowneighs, rowrank, colrank, colneighs;
    MPI_Comm_size(ColWorld, &colneighs);
    MPI_Comm_size(RowWorld, &rowneighs);
    MPI_Comm_rank(RowWorld, &rowrank);
    MPI_Comm_rank(ColWorld, &colrank);

    // int64_t maxnrow = std::max(perm_pvec_M11.Reduce(maximum<int64_t>(), 0.0) +1, M11.getnrow());
    // int64_t maxncol = std::max(perm_qvec_M11.Reduce(maximum<int64_t>(), 0.0) +1, M11.getnrow());
    float M11_load = M11.LoadImbalance();
    if(myrank == 0) printf("inside exapand and permute function load imbalance of input matrix before %f\n", M11_load);
    
    // int64_t max_pvec = perm_pvec_M11.Reduce(maximum<int64_t>(), 0.0);
    // int64_t max_qvec = perm_qvec_M11.Reduce(maximum<int64_t>(), 0.0);
    // int64_t M11_nrow = M11.getnrow();
    // int64_t M11_ncol = M11.getncol();

    MPI_Barrier(World);
    double start = MPI_Wtime();

    /// ***** perm_vec col gather ***** //
    int * localQVecSize = new int[nprocs];  //changed rowsize tp localVecSize
    localQVecSize[myrank] = perm_qvec_M11.LocArrSize();
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, localQVecSize, 1, MPI_INT, World);

    int * dpls = new int[nprocs]();	
    int * recvcount = new int[nprocs]();
    for (int i=rowrank*rowneighs; i<nprocs; i++)
    {
        if (i < rowrank*rowneighs+rowneighs)
            recvcount[i] = localQVecSize[i];
        if((i > rowrank*rowneighs) && (i < rowrank*rowneighs+rowneighs))
            dpls[i] = recvcount[i-1] +dpls[i-1];
        else if(i >= rowrank*rowneighs+rowneighs)
            dpls[i] = dpls[i-1];
    }
    
    int accsize_col = std::accumulate(recvcount, recvcount+nprocs, 0);
    int64_t * perm_vec_col = new int64_t[accsize_col]();
    int64_t * localp = const_cast<int64_t*>(SpHelper::p2a(perm_qvec_M11.GetLocVec()));
    int * senddpls = new int[nprocs](); 
    int * sendcounts = new int[nprocs](); 
    for (int i=0; i<rowneighs; i++)
        sendcounts[colrank+i*rowneighs] = localQVecSize[myrank];
    MPI_Alltoallv(localp, &sendcounts[0], senddpls, MPIType<int64_t>(), perm_vec_col, recvcount, dpls, MPIType<int64_t>(), World);
    /// ***** End of perm_vec col gather ***** //
    /// ***** perm_vec row gather ***** //
    int * localPVecSize = new int[nprocs];  //changed rowsize tp localVecSize
    localPVecSize[myrank] = perm_pvec_M11.LocArrSize();
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, localPVecSize, 1, MPI_INT, World);
    int * rowsize = new int[rowneighs];
    int * dpls_row = new int[rowneighs]();	// displacements (zero initialized pid) 
    #pragma omp parallel for
    for (int i=0; i<rowneighs; i++)
        rowsize[i] = localPVecSize[colrank*rowneighs+i];
    std::partial_sum(rowsize, rowsize+rowneighs-1, dpls_row+1);        
    int accsize = std::accumulate(rowsize, rowsize+rowneighs, 0);
    // printf("rank %d accsize %d\n", myrank, accsize);
    int64_t * perm_vec_row = new int64_t[accsize];
    int trxsize = perm_pvec_M11.LocArrSize();
    int64_t * trxnums = const_cast<int64_t*>(SpHelper::p2a(perm_pvec_M11.GetLocVec()));
    MPI_Allgatherv(trxnums, trxsize, MPIType<int64_t>(), perm_vec_row, rowsize, dpls_row, MPIType<int64_t>(), RowWorld);
    delete localQVecSize, localPVecSize, dpls, recvcount, senddpls, sendcounts, rowsize, dpls_row;
    // **************** End of perm_vec row gather **************** /// 
    //  *************  Go over nonzeros and prepare sendtuples ********** //
    Dcsc<int64_t, double>* Adcsc = (M11.seqptr())->GetDCSC();
    int64_t Anzc = 0, Annz = 0;
    if( Adcsc != NULL && Adcsc!= 0)
    {
        Anzc = Adcsc->nzc;
        Annz = Adcsc->nz;
    }
    std::vector<int64_t> thrdSegmentPntr(nthreads+1);
    thrdSegmentPntr[0] = static_cast<int64_t>(0);
    int64_t thrdSegNum = static_cast<int64_t> (0);
    int64_t segmentSize = Annz/nthreads;
    for (int64_t i=0; i<Anzc-1; i++)
    {
        if (*((Adcsc->cp)+i) > (thrdSegNum+1)*(segmentSize))
        {
            thrdSegNum++;
            if(thrdSegNum < nthreads)
                thrdSegmentPntr[thrdSegNum] = i;
            else
                thrdSegmentPntr[thrdSegNum] = static_cast<int64_t>(Anzc);
        }
    }
    for( int i=thrdSegNum+1; i<nthreads+1; i++)
        thrdSegmentPntr[i] = static_cast<int64_t>(Anzc);
    std::vector<std::vector<int64_t>> threadSendCounter(nthreads, std::vector<int64_t> (nprocs, static_cast<int64_t>(0)));
    std::vector<int64_t> localSendAccum(nprocs, 0);
    std::vector<vector<int64_t>> threadSendAccum(nthreads, localSendAccum);
    int64_t out_m_perproc = maxnrow / M11gridrows;   //num of rows per processor
    int64_t out_n_perproc = maxnrow / M11gridcols;
    // int64_t M11nrow = M11.getnrow();
    if (Adcsc != NULL && Adcsc!= 0)
    {
        #pragma omp parallel for schedule(static) num_threads(nthreads)
        for( int k=0; k<nthreads; k++)
        {
            int myThread = omp_get_thread_num();
            for(int64_t i=thrdSegmentPntr[k]; i < thrdSegmentPntr[k+1]; i++)
            {
                for(int64_t j = *((Adcsc->cp)+i); j< *((Adcsc->cp)+i+1); j++)
                {
                    int64_t currlocalrow = *((Adcsc->ir)+j);
                    int64_t currlocalcol = *((Adcsc->jc)+i);
                    if(currlocalcol < accsize_col && currlocalrow < accsize) {
                        int64_t newgrow = perm_vec_row[currlocalrow];
                        int64_t newgcol = perm_vec_col[currlocalcol];
                        int64_t newcolrank2 = std::min(newgcol / out_n_perproc, static_cast<int64_t>(M11gridcols-1));
                        int64_t newrowrank2 = std::min(newgrow / out_m_perproc, static_cast<int64_t>(M11gridrows-1));
                        int64_t targetproc = static_cast<int64_t>(M11commGrid->GetRank(static_cast<int>(newrowrank2), static_cast<int>(newcolrank2))); 
                        threadSendCounter[myThread][targetproc]++;
                    }
                }
            }
        }
    }
    int * sendcnt = new int[nprocs]();
    int * recvcnt = new int[nprocs];
    int * sdispls = new int[nprocs]();
    int * rdispls = new int[nprocs]();
    #pragma omp parallel for 
    for( int j=0 ; j<nprocs; j++)
    {
        threadSendAccum[0][j] = 0;
        sendcnt[j] += threadSendCounter[0][j];
        for( int i=1; i<nthreads; i++)
        {
            threadSendAccum[i][j] = threadSendCounter[i-1][j] + threadSendAccum[i-1][j];
            sendcnt[j] += threadSendCounter[i][j];
        }
    }  
    // Create/allocate variables for vector assignment 
    MPI_Datatype MPI_tuple;
    MPI_Type_contiguous(sizeof(std::tuple<int64_t,int64_t,double>), MPI_CHAR, &MPI_tuple);
    MPI_Type_commit(&MPI_tuple);
        
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);

    std::partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
    std::partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
    int64_t totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<int64_t>(0));
    std::vector< std::tuple<int64_t,int64_t,double> > sendTuples(Annz);
    if (Adcsc != NULL && Adcsc!= 0)
    {
        #pragma omp parallel for schedule(static) num_threads(nthreads)
        for( int k=0; k<nthreads; k++)
        {
            std::vector<int64_t> fillCounter(nprocs, static_cast<int64_t>(0));
            int myThread = omp_get_thread_num();
            for(int64_t i=thrdSegmentPntr[k]; i<thrdSegmentPntr[k+1]; i++)
            {
                for(int64_t j = *((Adcsc->cp)+i); j< *((Adcsc->cp)+i+1); j++)
                {
                    double tempval = * ((Adcsc->numx)+j) ;
                    int64_t currlocalrow = *((Adcsc->ir)+j);
                    int64_t currlocalcol = *((Adcsc->jc)+i);
                    if(currlocalcol < accsize_col && currlocalrow < accsize) {
                        int64_t currgrow = currlocalrow + (maxnrow / M11gridrows) * colrank;
                        int64_t currgcol = currlocalcol + (maxncol / M11gridcols) * rowrank;
                        int64_t newgrow = perm_vec_row[currlocalrow];
                        int64_t newgcol = perm_vec_col[currlocalcol];
                        int64_t newcolrank2 = std::min(newgcol / out_n_perproc, static_cast<int64_t>(M11gridcols-1));
                        int64_t newrowrank2 = std::min(newgrow / out_m_perproc, static_cast<int64_t>(M11gridrows-1));
                        int64_t newlrow2 = newgrow - (newrowrank2 * out_n_perproc) ;
                        int64_t newlcol2 = newgcol - (newcolrank2 * out_m_perproc) ;
                        int64_t targetproc = static_cast<int64_t>(M11commGrid->GetRank(static_cast<int>(newrowrank2), static_cast<int>(newcolrank2))); 
                        sendTuples[static_cast<size_t>(sdispls[targetproc]+threadSendAccum[myThread][targetproc]+fillCounter[targetproc])] = std::make_tuple(newlrow2, newlcol2, tempval);
                        fillCounter[targetproc]++;
                    }
                    
                }
            }
        }
    }
    std::vector< std::tuple<int64_t,int64_t,double> > recvTuples(totrecv);
    MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples.data(), recvcnt, rdispls, MPI_tuple, World);

    Dist::DCCols* newspSeq;
    int64_t localnrow = (colrank == M11gridcols-1) ? (maxnrow + M11gridcols - 1) / M11gridcols : maxnrow / M11gridcols ;
    int64_t localncol = (rowrank == M11gridrows-1) ? (maxnrow + M11gridrows - 1) / M11gridrows : maxnrow / M11gridrows ;
    if(recvTuples.size() > 0)
    {   
        int recvTupleSize = recvTuples.size();
        int chunkNums = 4*nthreads;
        if( nthreads*2 < recvTupleSize && recvTupleSize < chunkNums*2)
            chunkNums = nthreads;
        else if(recvTupleSize <= nthreads*2 )
            chunkNums = 1;
        vector<SpTuples<int64_t,double>*> ArrSpTups(chunkNums);
        int64_t chunkSize = std::floor(recvTupleSize / chunkNums);

        #pragma omp parallel
        #pragma omp for schedule(dynamic)
        for (int i=0; i<chunkNums; i++)
        {
            int64_t myChunkSize = (i==chunkNums-1) ? (recvTupleSize - (chunkNums-1)*chunkSize) : chunkSize;
            ArrSpTups[i] = new SpTuples<int64_t, double>(myChunkSize, localnrow, localncol, &(recvTuples[chunkSize*i]), false, false);
        }
        SpTuples<int64_t,double> * sortedTuples = MultiwayMerge<PTFF>(ArrSpTups,localnrow, localncol, false);
        // SpTuples<int64_t,double> * sortedTuples = MultiwayMergeHash<PTFF>(ArrSpTups,localnrow, localncol);
        newspSeq = new Dist::DCCols(*sortedTuples, false);
        for(int i = 0; i < chunkNums; i++){
            ArrSpTups[i]->tuples_deleted = true; // Temporary patch to avoid memory leak and segfault
            delete ArrSpTups[i];
        }
        delete sortedTuples;
    }
    else
    {
        newspSeq = new Dist::DCCols(static_cast<int64_t>(0), localnrow, localncol, static_cast<int64_t>(0));
    }
    // Dist::MPI_DCCols Astar(newspSeq, M11commGrid) ;
    MPI_Barrier(World);
    
    double end = MPI_Wtime();
    if(myrank == 0) printf("expand permute matrix in function %f\n", end-start);
    
    fflush(stdout);
    MPI_Barrier(World);
    Dist::MPI_DCCols M11_new(newspSeq, M11commGrid);
    M11.FreeMemory();
    if(myrank == 0) printf("M11 new created \n");
    fflush(stdout);
    M11_new.PrintInfo();
    M11 = M11_new;
    if(myrank == 0) printf("M11 copied over\n");
    fflush(stdout);
    M11.PrintInfo();
    M11_load = M11.LoadImbalance();
    if(myrank == 0) printf("inside expand and permute function load imbalance of input matrix after expand and permutation %f\n", M11_load);

    delete[] sendcnt;
    delete[] recvcnt;
    delete[] sdispls;
    delete[] rdispls;
    MPI_Type_free(&MPI_tuple);
    

    return 0;
}




int MatrixPermute(Dist::MPI_DCCols &M11, MPI_Comm World) {

// int ExpandPermute(Dist::MPI_DCCols &M11, FullyDistVec<int, int64_t> &perm_pvec_M11, FullyDistVec<int, int64_t> &perm_qvec_M11, int64_t maxnrow, int64_t maxncol, MPI_Comm World) {
    
    int nthreads = 1;
#ifdef THREADED
#pragma omp parallel
{
    nthreads = omp_get_num_threads();
}
#endif
    // Define types for combBLAS
    typedef PlusTimesSRing<double, double> PTFF;

    // double rsstrt, rsend; 
    // MPI_Barrier(World);
    // rsstrt = MPI_Wtime();
    
    int nprocs, myrank;
    MPI_Comm_size(World,&nprocs);
    MPI_Comm_rank(World, &myrank);
    auto M11commGrid = M11.getcommgrid();
    int M11gridrows = M11commGrid->GetGridRows();
    int M11gridcols = M11commGrid->GetGridCols();
    MPI_Comm ColWorld = M11commGrid->GetColWorld();
    MPI_Comm RowWorld = M11commGrid->GetRowWorld();

    int rowneighs, rowrank, colrank, colneighs;
    MPI_Comm_size(ColWorld, &colneighs);
    MPI_Comm_size(RowWorld, &rowneighs);
    MPI_Comm_rank(RowWorld, &rowrank);
    MPI_Comm_rank(ColWorld, &colrank);

    // int64_t maxnrow = std::max(perm_pvec_M11.Reduce(maximum<int64_t>(), 0.0) +1, M11.getnrow());
    // int64_t maxncol = std::max(perm_qvec_M11.Reduce(maximum<int64_t>(), 0.0) +1, M11.getnrow());
    float M11_load = M11.LoadImbalance();
    if(myrank == 0) printf("inside permute function load imbalance of input matrix before permutation %f\n", M11_load);
    M11.PrintInfo();
    int64_t maxnrow = M11.getnrow();
    int64_t maxncol = M11.getncol();

    MPI_Barrier(World);
    double start = MPI_Wtime();

    FullyDistVec<int, int64_t> perm_pvec_M11(M11commGrid);
    perm_pvec_M11.iota(maxnrow, 0);
    perm_pvec_M11.RandPerm();
    FullyDistVec<int, int64_t> perm_qvec_M11(M11commGrid);
    perm_qvec_M11.iota(maxncol, 0);
    perm_qvec_M11.RandPerm();
    ExpandPermute(M11, perm_pvec_M11, perm_pvec_M11, maxnrow, maxncol, World);

    M11.PrintInfo();
    fflush(stdout);
    M11_load = M11.LoadImbalance();
    if(myrank == 0) printf("inside permute function load imbalance of input matrix after permutation %f\n", M11_load);
    fflush(stdout);


    return 0;
}


// Function to perform matrix assignment,
int MatrixAssignment(Dist::MPI_DCCols &A, Dist::MPI_DCCols &B, FullyDistVec<int, int64_t> &rvec, FullyDistVec<int, int64_t> &cvec, bool pruneA, MPI_Comm World) {
    
    int nthreads = 1;
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    
    int nprocs, myrank;
    MPI_Comm_size(World,&nprocs);
    MPI_Comm_rank(World, &myrank);
    // Define types for combBLAS
    typedef PlusTimesSRing<double, double> PTFF;

    // Setup communicators
    auto AcommGrid = A.getcommgrid();
    int Agridrows = AcommGrid->GetGridRows();
    int Agridcols = AcommGrid->GetGridCols();
    MPI_Comm ColWorld = AcommGrid->GetColWorld();
    MPI_Comm RowWorld = AcommGrid->GetRowWorld();
    int rowneighs, rowrank, colrank;
    MPI_Comm_size(RowWorld, &rowneighs);
    MPI_Comm_rank(RowWorld, &rowrank);
    MPI_Comm_rank(ColWorld, &colrank);

    double tstart, tend; 

    // **************** Start of Local col vec gathering ******************** //
    int * localVecSize = new int[nprocs];  //changed rowsize tp localVecSize
    localVecSize[myrank] = cvec.LocArrSize();

    MPI_Barrier(World);
    tstart = MPI_Wtime();
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, localVecSize, 1, MPI_INT, World);
    MPI_Barrier(World);
    double comm_1_end = MPI_Wtime();
    
    int * dpls = new int[nprocs](); // displacements (zero initialized pid) 
    int * recvcount = new int[nprocs]();
    for (int i=rowrank*rowneighs; i<nprocs; i++)
    {
        if (i < rowrank*rowneighs+rowneighs)
            recvcount[i] = localVecSize[i];
        if((i > rowrank*rowneighs) && (i < rowrank*rowneighs+rowneighs))
            dpls[i] = recvcount[i-1] +dpls[i-1];
        else if(i >= rowrank*rowneighs+rowneighs)
            dpls[i] = dpls[i-1];
    }
    
    int64_t accsize_col = std::accumulate(recvcount, recvcount+nprocs, 0);
    int64_t * numacc_col = new int64_t[accsize_col]();
    
    int64_t * localp = const_cast<int64_t*>(SpHelper::p2a(cvec.GetLocVec()));

    int * senddpls = new int[nprocs](); 
    int * sendcounts = new int[nprocs](); 
    for (int i=0; i<rowneighs; i++)
        sendcounts[colrank+i*rowneighs] = localVecSize[myrank];
    MPI_Barrier(World);
    double comm_2_bgn = MPI_Wtime();
    MPI_Alltoallv(localp, &sendcounts[0], senddpls, MPIType<int64_t>(), numacc_col, recvcount, dpls, MPIType<int64_t>(), World);
    MPI_Barrier(World);
    double comm_2_end = MPI_Wtime();
    // **************** End of my local col vec gathering **************** /// 

    // **************** Start of my local row vec gathering **************** /// 
    int * rowsize = new int[rowneighs];
    int * dpls_row = new int[rowneighs]();  // displacements (zero initialized pid) 

    // #pragma omp parallel for
    for (int i=0; i<rowneighs; i++)
        rowsize[i] = localVecSize[colrank*rowneighs+i];
    
    std::partial_sum(rowsize, rowsize+rowneighs-1, dpls_row+1);        
    int64_t accsize = std::accumulate(rowsize, rowsize+rowneighs, 0);
    int64_t * numacc_row = new int64_t[accsize];
    int64_t trxsize = rvec.LocArrSize();
    int64_t * trxnums = const_cast<int64_t*>(SpHelper::p2a(rvec.GetLocVec()));
        
    MPI_Barrier(World);
    double comm_3_bgn = MPI_Wtime();
    MPI_Allgatherv(trxnums, trxsize, MPIType<int64_t>(), numacc_row, rowsize, dpls_row, MPIType<int64_t>(), RowWorld);
    MPI_Barrier(World);
    double comm_3_end = MPI_Wtime();
    // **************** End of my local row vec gathering **************** /// 
    
    Dist::DCCols *BspSeq = B.seqptr();
    Dcsc<int64_t, double>* Bdcsc = BspSeq->GetDCSC();
    int64_t Bnzc = Bdcsc->nzc;
    std::vector<int64_t> thrdSegmentPntr(nthreads+1);
    thrdSegmentPntr[0] = static_cast<int64_t>(0);
    int64_t thrdSegNum = static_cast<int64_t> (0);
    int64_t segmentSize = (Bdcsc->nz)/nthreads;
    
    for (int64_t i=0; i<Bnzc-1; i++)
    {
        if (*((Bdcsc->cp)+i) > (thrdSegNum+1)*(segmentSize))
        {
            thrdSegNum++;
            if(thrdSegNum < nthreads)
                thrdSegmentPntr[thrdSegNum] = i;
            else
                thrdSegmentPntr[thrdSegNum] = static_cast<int64_t>(Bnzc);
        }
    }
    for( int i=thrdSegNum+1; i<nthreads+1; i++)
        thrdSegmentPntr[i] = static_cast<int64_t>(Bnzc);

    std::vector<std::vector<int64_t>> threadSendCounter(nthreads, std::vector<int64_t> (nprocs, static_cast<int64_t>(0)));
    std::vector<int64_t> localSendAccum(nprocs, 0);
    std::vector<vector<int64_t>> threadSendAccum(nthreads, localSendAccum);
    
    int64_t out_m_perproc = A.getnrow() / Agridrows;   //num of rows per processor
    int64_t out_n_perproc = A.getncol() / Agridcols;
    MPI_Barrier(World);
    double frst_for_bgn = MPI_Wtime();
    if (Bdcsc != NULL && Bdcsc!= 0)
    {
        #pragma omp parallel for schedule(static) num_threads(nthreads)
        for( int64_t k=0; k<nthreads; k++)
        {
            int newrowrank ;
            int newcolrank ;
            int myThread = omp_get_thread_num();
            for(int64_t i=thrdSegmentPntr[k]; i<thrdSegmentPntr[k+1]; i++)
            {
                for(int64_t j = *((Bdcsc->cp)+i); j< *((Bdcsc->cp)+i+1); j++)
                {
                    int currlocalrow = *((Bdcsc->ir)+j);
                    int currlocalcol = *((Bdcsc->jc)+i);
                    if(out_n_perproc != 0)
                        newcolrank = std::min(static_cast<int>(numacc_col[currlocalcol] / out_n_perproc), Agridcols-1);
                    else    // all owned by the last processor row
                        newcolrank = Agridcols -1;
                    if(out_m_perproc != 0)
                        newrowrank = std::min(static_cast<int>(numacc_row[currlocalrow] / out_m_perproc), Agridrows-1);
                    else    // all owned by the last processor row
                        newrowrank = Agridrows -1;
                    int targetproc = AcommGrid->GetRank(newrowrank, newcolrank); 
                    threadSendCounter[myThread][targetproc]++;
                }
            }
        }
    }
    MPI_Barrier(World);
    double scnd_for_bgn = MPI_Wtime();

    int * sendcnt = new int[nprocs]();
    int * recvcnt = new int[nprocs];
    int * sdispls = new int[nprocs]();
    int * rdispls = new int[nprocs]();
    
    #pragma omp parallel for 
    for( int j=0 ; j<nprocs; j++)
    {
        threadSendAccum[0][j] = 0;
        sendcnt[j] += threadSendCounter[0][j];
        for( int i=1; i<nthreads; i++)
        {
            threadSendAccum[i][j] = threadSendCounter[i-1][j] + threadSendAccum[i-1][j];
            sendcnt[j] += threadSendCounter[i][j];
        }

    }  

    // Create/allocate variables for vector assignment 
    MPI_Datatype MPI_tuple;
    MPI_Type_contiguous(sizeof(std::tuple<int, int, double>), MPI_CHAR, &MPI_tuple);
    MPI_Type_commit(&MPI_tuple);
    
    MPI_Barrier(World);
    double comm_4_bgn = MPI_Wtime();   
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
    MPI_Barrier(World);
    double comm_4_end = MPI_Wtime();

    std::partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
    std::partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
    int64_t totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<int64_t>(0));
    int64_t totsend = Bdcsc->nz;
    MPI_Barrier(World);
    double third_for_bgn = MPI_Wtime();
    std::vector< std::tuple<int, int, double> > sendTuples(totsend);
    if (Bdcsc != NULL && Bdcsc!= 0)
    {
        
        #pragma omp parallel for schedule(static) num_threads(nthreads)
        for( int k=0; k<nthreads; k++)
        {
            int newrowrank ;
            int newcolrank ;
            std::vector<int64_t> fillCounter(nprocs, static_cast<int64_t>(0));
            int myThread = omp_get_thread_num();
            for(int i=thrdSegmentPntr[k]; i<thrdSegmentPntr[k+1]; i++)
            {
                for(int j = *((Bdcsc->cp)+i); j< *((Bdcsc->cp)+i+1); j++)
                {

                    double tempval = * ((Bdcsc->numx)+j) ;
                    int currlocalrow = *((Bdcsc->ir)+j);
                    int currlocalcol = *((Bdcsc->jc)+i);
                    if(out_n_perproc != 0)
                        newcolrank = std::min(static_cast<int>(numacc_col[currlocalcol] / out_n_perproc), Agridcols-1);
                    else    // all owned by the last processor row
                        newcolrank = Agridcols -1;
                    if(out_m_perproc != 0)
                        newrowrank = std::min(static_cast<int>(numacc_row[currlocalrow] / out_m_perproc), Agridrows-1);
                    else    // all owned by the last processor row
                        newrowrank = Agridrows -1;
                    int targetproc = AcommGrid->GetRank(newrowrank, newcolrank); 
                    int newlcol = numacc_col[currlocalcol] - (newcolrank * out_n_perproc) ;
                    int newlrow = numacc_row[currlocalrow] - (newrowrank * out_m_perproc) ;
                    sendTuples[sdispls[targetproc]+threadSendAccum[myThread][targetproc]+fillCounter[targetproc]] = std::make_tuple(newlrow, newlcol, tempval);
                    fillCounter[targetproc]++;
                }
            }
        }
    }
    
    MPI_Barrier(World);
    double forth_for_bgn = MPI_Wtime();
    std::vector< std::tuple<int64_t,int64_t,double> > recvTuples(totrecv);

    MPI_Barrier(World);
    double comm_5_bgn = MPI_Wtime();
    MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples.data(), recvcnt, rdispls, MPI_tuple, World);
    MPI_Barrier(World);
    double comm_5_end = MPI_Wtime();

    int Anrow = A.getnrow();
    int Ancol = A.getncol();

    Dist::DCCols *AspSeq = A.seqptr();

    Dist::DCCols* newspSeq;
    MPI_Barrier(World);
    double loc_newb = MPI_Wtime();
    if(recvTuples.size() > 0)
    {   

        int recvTupleSize = recvTuples.size();
        int chunkNums = 4*nthreads;
        if( nthreads*2 < recvTupleSize && recvTupleSize < chunkNums*2)
            chunkNums = nthreads;
        else if(recvTupleSize <= nthreads*2 )
            chunkNums = 1;
        vector<SpTuples<int64_t,double>*> ArrSpTups(chunkNums);
        int64_t chunkSize = std::floor(recvTupleSize / chunkNums);

        #pragma omp parallel
        #pragma omp for schedule(dynamic)
        for (int i=0; i<chunkNums; i++)
        {
            int64_t myChunkSize = (i==chunkNums-1) ? (recvTupleSize - (chunkNums-1)*chunkSize) : chunkSize;
            ArrSpTups[i] = new SpTuples<int64_t, double>(myChunkSize, AspSeq->getnrow(), AspSeq->getncol(), &(recvTuples[chunkSize*i]), false, false);
        }
        int64_t ArrSpTups_zize = 0; 
        for(int k=0; k<chunkNums; k++)
            ArrSpTups_zize += ArrSpTups[k]->getnnz();
        int64_t sortedtuple_size = ArrSpTups.size();    
        SpTuples<int64_t,double> * sortedTuples = MultiwayMerge<PTFF>(ArrSpTups, AspSeq->getnrow(), AspSeq->getncol(), false);
        newspSeq = new Dist::DCCols(*sortedTuples, false);
        for(int i = 0; i < chunkNums; i++){
            ArrSpTups[i]->tuples_deleted = true; 
            delete ArrSpTups[i];
        }
    }
    else
    {
        newspSeq = new Dist::DCCols(static_cast<int>(0), AspSeq->getnrow(), AspSeq->getncol(), static_cast<int>(0));
    }
    MPI_Barrier(World);
    Dist::MPI_DCCols newB(newspSeq, AcommGrid) ;
    
    if(myrank == 0) printf("newB created\n");
    newB.PrintInfo();

    MPI_Barrier(World);
    double loc_prune = MPI_Wtime();
    if(pruneA == true)
    {
        FullyDistSpVec<int, int64_t> sprvec(rvec, [](int64_t el){return el;});
        sprvec = sprvec.Invert(Anrow);
        sprvec.Apply([](int64_t el){return 1;});
        FullyDistVec<int, int64_t> denseRi(sprvec);
        FullyDistSpVec<int, int64_t> spcvec(cvec, [](int64_t el){return el;});
        spcvec = spcvec.Invert(Ancol);
        spcvec.Apply([](int64_t el){return 1;});
        FullyDistVec<int, int64_t> denseCi(spcvec);
        Dist::MPI_DCCols Mask(A);
        Mask.DimApply(Row, denseRi, [](float mv, float vv){return vv;});
        Mask.PruneColumn(denseCi, [](float mv, float vv){return static_cast<float>((vv * mv)!=1);}, true);
        A.SetDifference(Mask);   
    }
    else if(pruneA == false)
    {
        A.SetDifference(newB);
    }
    if(myrank == 0) printf("after A pruned/setdifference \n");
    newB.PrintInfo();
    A.PrintInfo();

    MPI_Barrier(World);
    double local_add = MPI_Wtime();
    A += newB; 
    MPI_Barrier(World);
    tend = MPI_Wtime();
    if(myrank == 0) printf("After adding newB to A \n");
    A.PrintInfo();
    if(myrank == 0)
        printf("1st for count: %f, 2nd for threadaccum: %f 3rd for sendtuples: %f local newB:%f prune A: %f A+newB:%f\n", scnd_for_bgn-frst_for_bgn, third_for_bgn-scnd_for_bgn, forth_for_bgn-third_for_bgn, loc_prune-loc_newb, local_add-loc_prune, tend-local_add);

    double communication_time = comm_1_end - tstart + comm_2_end - comm_2_bgn + comm_3_end - comm_3_bgn + comm_4_end - comm_4_bgn + comm_5_end - comm_5_bgn;
    if(myrank == 0) printf( "my assign time is %f\n", tend - tstart );  
    if(myrank == 0) printf("my assign communication time:%f  \n End of Matrix Assignment \n", communication_time);
    fflush(stdout);
    return 0;
}


void MatrixAssignmentWithPermute(Dist::MPI_DCCols &A, Dist::MPI_DCCols &B, FullyDistVec<int, int64_t> &rvec, FullyDistVec<int, int64_t> &cvec, bool pruneA, MPI_Comm World) {
    
    int nthreads = 1;
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    
    int nprocs, myrank;
    MPI_Comm_size(World,&nprocs);
    MPI_Comm_rank(World, &myrank);
    typedef PlusTimesSRing<double, double> PTFF;

    // Setup communicators
    auto AcommGrid = A.getcommgrid();
    int Agridrows = AcommGrid->GetGridRows();
    int Agridcols = AcommGrid->GetGridCols();
    MPI_Comm ColWorld = AcommGrid->GetColWorld();
    MPI_Comm RowWorld = AcommGrid->GetRowWorld();
    int rowneighs, rowrank, colrank, colneighs;
    MPI_Comm_size(ColWorld, &colneighs);
    MPI_Comm_size(RowWorld, &rowneighs);
    MPI_Comm_rank(RowWorld, &rowrank);
    MPI_Comm_rank(ColWorld, &colrank);

    fflush(stdout);
    int64_t maxnrow = std::max(rvec.Reduce(maximum<int64_t>(), 0.0) +1, A.getnrow());
    int64_t maxncol = std::max(cvec.Reduce(maximum<int64_t>(), 0.0) +1, A.getncol());
    maxnrow = maxncol = std::max(maxncol, maxnrow);
    if(myrank == 0 && maxnrow != maxncol)   printf("Error - Input Graph is not square\n");
    if(myrank == 0)     printf("result matrices are %d x %d \n", maxnrow, maxncol);
    //  ***** permutation vector *** //
    FullyDistVec<int, int64_t> perm_pvec(AcommGrid);
    perm_pvec.iota(maxnrow, 0);
    perm_pvec.RandPerm();
    FullyDistVec<int, int64_t> M11_pidx(AcommGrid);
    FullyDistVec<int, int64_t> M11_qidx(AcommGrid);
    M11_pidx.iota( A.getnrow(), 0);
    M11_qidx.iota( A.getncol(), 0);
    FullyDistVec<int, int64_t> perm_pvec_M11 = perm_pvec(M11_pidx);
    FullyDistVec<int, int64_t> perm_qvec_M11 = perm_pvec(M11_qidx);

    //Expand and permute A
    MPI_Barrier(World);
    double tassignSt = MPI_Wtime();
    ExpandPermute(A, perm_pvec_M11, perm_qvec_M11, maxnrow, maxncol, World);
    MPI_Barrier(World);
    double tassignEnd = MPI_Wtime();
    if(myrank == 0)     printf("time for expand and permute A : %f\n", tassignEnd-tassignSt);


    rvec = perm_pvec(rvec);
    cvec = perm_pvec(cvec);
    // **************** Start of Local col vec gathering ******************** //
    int * localVecSize = new int[nprocs];  
    localVecSize[myrank] = cvec.LocArrSize();
    MPI_Barrier(World);
    double tstart = MPI_Wtime();
    MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, localVecSize, 1, MPI_INT, World);
    MPI_Barrier(World);
    double comm_1_end = MPI_Wtime();

    int * dpls = new int[nprocs]();	// displacements (zero initialized pid) 
    int * recvcount = new int[nprocs]();
    for (int i=rowrank*rowneighs; i<nprocs; i++)
    {
        if (i < rowrank*rowneighs+rowneighs)
            recvcount[i] = localVecSize[i];
        if((i > rowrank*rowneighs) && (i < rowrank*rowneighs+rowneighs))
            dpls[i] = recvcount[i-1] +dpls[i-1];
        else if(i >= rowrank*rowneighs+rowneighs)
            dpls[i] = dpls[i-1];
    }

    int64_t accsize_col = std::accumulate(recvcount, recvcount+nprocs, 0);
    int64_t * numacc_col = new int64_t[accsize_col]();
    
    int64_t * localp = const_cast<int64_t*>(SpHelper::p2a(cvec.GetLocVec()));

    int * senddpls = new int[nprocs](); 
    int * sendcounts = new int[nprocs](); 
    for (int i=0; i<rowneighs; i++)
        sendcounts[colrank+i*rowneighs] = localVecSize[myrank];
    MPI_Barrier(World);
    double comm_2_bgn = MPI_Wtime();
    MPI_Alltoallv(localp, &sendcounts[0], senddpls, MPIType<int64_t>(), numacc_col, recvcount, dpls, MPIType<int64_t>(), World);
    MPI_Barrier(World);
    double comm_2_end = MPI_Wtime();
    delete[] dpls, recvcount, senddpls, sendcounts;
    // **************** End of my local col vec gathering **************** /// 

    // **************** Start of my local row vec gathering **************** /// 
    int * rowsize = new int[rowneighs];
    int * dpls_row = new int[rowneighs]();	// displacements (zero initialized pid) 

    // #pragma omp parallel for
    for (int i=0; i<rowneighs; i++)
        rowsize[i] = localVecSize[colrank*rowneighs+i];
    
    std::partial_sum(rowsize, rowsize+rowneighs-1, dpls_row+1);        
    int64_t accsize = std::accumulate(rowsize, rowsize+rowneighs, 0);
    int64_t * numacc_row = new int64_t[accsize];
    int64_t trxsize = rvec.LocArrSize();
    int64_t * trxnums = const_cast<int64_t*>(SpHelper::p2a(rvec.GetLocVec()));
    
    double comm_3_bgn = MPI_Wtime();
    MPI_Allgatherv(trxnums, trxsize, MPIType<int64_t>(), numacc_row, rowsize, dpls_row, MPIType<int64_t>(), RowWorld);
    MPI_Barrier(World);
    double comm_3_end = MPI_Wtime();
    delete[] rowsize, dpls_row, localVecSize;
    // **************** End of my local row vec gathering **************** /// 
  
    Dist::DCCols *BspSeq = B.seqptr();
    Dcsc<int64_t, double>* Bdcsc = BspSeq->GetDCSC();
    int64_t Bnzc = Bdcsc->nzc;
    std::vector<int64_t> thrdSegmentPntr(nthreads+1);
    thrdSegmentPntr[0] = static_cast<int64_t>(0);
    int64_t thrdSegNum = static_cast<int64_t> (0);
    int64_t segmentSize = (Bdcsc->nz)/nthreads;
    
    for (int64_t i=0; i<Bnzc-1; i++)
    {
        if (*((Bdcsc->cp)+i) > (thrdSegNum+1)*(segmentSize))
        {
            thrdSegNum++;
            if(thrdSegNum < nthreads)
                thrdSegmentPntr[thrdSegNum] = i;
            else
                thrdSegmentPntr[thrdSegNum] = static_cast<int64_t>(Bnzc);
        }
    }
    for( int i=thrdSegNum+1; i<nthreads+1; i++)
        thrdSegmentPntr[i] = static_cast<int64_t>(Bnzc);

    std::vector<std::vector<int64_t>> threadSendCounter(nthreads, std::vector<int64_t> (nprocs, static_cast<int64_t>(0)));
    std::vector<int64_t> localSendAccum(nprocs, 0);
    std::vector<vector<int64_t>> threadSendAccum(nthreads, localSendAccum);
    
    // int64_t out_m_perproc = A.getnrow() / Agridrows;   //num of rows per processor
    // int64_t out_n_perproc = A.getncol() / Agridcols;
    int64_t out_m_perproc = maxnrow / Agridrows;   //num of rows per processor
    int64_t out_n_perproc = maxncol / Agridcols;
    
    MPI_Barrier(World);
    double frst_for_bgn = MPI_Wtime();
    if (Bdcsc != NULL && Bdcsc!= 0)
    {
        #pragma omp parallel for schedule(static) num_threads(nthreads)
        for( int64_t k=0; k<nthreads; k++)
        {
            int newrowrank ;
            int newcolrank ;
            int myThread = omp_get_thread_num();
            for(int64_t i=thrdSegmentPntr[k]; i<thrdSegmentPntr[k+1]; i++)
            {
                for(int64_t j = *((Bdcsc->cp)+i); j< *((Bdcsc->cp)+i+1); j++)
                {
                    int currlocalrow = *((Bdcsc->ir)+j);
                    int currlocalcol = *((Bdcsc->jc)+i);
                    if(out_n_perproc != 0)
                        newcolrank = std::min(static_cast<int>(numacc_col[currlocalcol] / out_n_perproc), Agridcols-1);
                    else	// all owned by the last processor row
                        newcolrank = Agridcols -1;
                    if(out_m_perproc != 0)
                        newrowrank = std::min(static_cast<int>(numacc_row[currlocalrow] / out_m_perproc), Agridrows-1);
                    else	// all owned by the last processor row
                        newrowrank = Agridrows -1;
                    int targetproc = AcommGrid->GetRank(newrowrank, newcolrank); 
                    threadSendCounter[myThread][targetproc]++;
                }
            }
        }
    }
    MPI_Barrier(World);
    double scnd_for_bgn = MPI_Wtime();
    
    int * sendcnt = new int[nprocs]();
    int * recvcnt = new int[nprocs];
    int * sdispls = new int[nprocs]();
    int * rdispls = new int[nprocs]();
    
    #pragma omp parallel for 
    for( int j=0 ; j<nprocs; j++)
    {
        threadSendAccum[0][j] = 0;
        sendcnt[j] += threadSendCounter[0][j];
        for( int i=1; i<nthreads; i++)
        {
            threadSendAccum[i][j] = threadSendCounter[i-1][j] + threadSendAccum[i-1][j];
            sendcnt[j] += threadSendCounter[i][j];
        }
    }  
    
    // Create/allocate variables for vector assignment 
    MPI_Datatype MPI_tuple;
    MPI_Type_contiguous(sizeof(std::tuple<int64_t,int64_t, double>), MPI_CHAR, &MPI_tuple);
    MPI_Type_commit(&MPI_tuple);
    
    MPI_Barrier(World);
    double comm_4_bgn = MPI_Wtime();   
    MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
    MPI_Barrier(World);
    double comm_4_end = MPI_Wtime();

    std::partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
    std::partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
    int64_t totrecv = accumulate(recvcnt,recvcnt+nprocs, static_cast<int64_t>(0));
    int64_t totsend = Bdcsc->nz;
    MPI_Barrier(World);
    double third_for_bgn = MPI_Wtime();
    std::vector< std::tuple<int64_t,int64_t, double> > sendTuples(totsend);
    if (Bdcsc != NULL && Bdcsc!= 0)
    {
        
        #pragma omp parallel for schedule(static) num_threads(nthreads)
        for( int k=0; k<nthreads; k++)
        {
            int newrowrank ;
            int newcolrank ;
            std::vector<int64_t> fillCounter(nprocs, static_cast<int64_t>(0));
            int myThread = omp_get_thread_num();
            for(int i=thrdSegmentPntr[k]; i<thrdSegmentPntr[k+1]; i++)
            {
                for(int j = *((Bdcsc->cp)+i); j< *((Bdcsc->cp)+i+1); j++)
                {

                    double tempval = * ((Bdcsc->numx)+j) ;
                    int currlocalrow = *((Bdcsc->ir)+j);
                    int currlocalcol = *((Bdcsc->jc)+i);
                    if(out_n_perproc != 0)
                        newcolrank = std::min(static_cast<int>(numacc_col[currlocalcol] / out_n_perproc), Agridcols-1);
                    else	// all owned by the last processor row
                        newcolrank = Agridcols -1;
                    if(out_m_perproc != 0)
                        newrowrank = std::min(static_cast<int>(numacc_row[currlocalrow] / out_m_perproc), Agridrows-1);
                    else	// all owned by the last processor row
                        newrowrank = Agridrows -1;
                
                    int targetproc = AcommGrid->GetRank(newrowrank, newcolrank); 
                    int newlcol = numacc_col[currlocalcol] - (newcolrank * out_n_perproc) ;
                    int newlrow = numacc_row[currlocalrow] - (newrowrank * out_m_perproc) ;
                    sendTuples[sdispls[targetproc]+threadSendAccum[myThread][targetproc]+fillCounter[targetproc]] = std::make_tuple(newlrow, newlcol, tempval);
                    fillCounter[targetproc]++;
                }
            }
        }
    }
    
    double forth_for_bgn = MPI_Wtime();
    std::vector< std::tuple<int64_t,int64_t,double> > recvTuples(totrecv);
    // std::vector< std::tuple<int, int, double> > recvTuples(totrecv);

    // printf("before alltoall myrank %d sendcnt : %d, recvcnt: %d\n", myrank, sendcnt[myrank], recvcnt[myrank]);
    // fflush(stdout);

    MPI_Barrier(World);
    double comm_5_bgn = MPI_Wtime();
    MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples.data(), recvcnt, rdispls, MPI_tuple, World);
    MPI_Barrier(World);
    double comm_5_end = MPI_Wtime();

    // int nrow = BspSeq->getnrow() ;
    // int ncol = BspSeq->getncol();
    
    delete[] sendcnt, recvcnt, sdispls, rdispls;
    MPI_Type_free(&MPI_tuple);
    //********
    Dist::DCCols *AspSeq = A.seqptr();

    Dist::DCCols* newspSeq;
    MPI_Barrier(World);
    double loc_newb = MPI_Wtime();
    if(recvTuples.size() > 0)
    {   

        int recvTupleSize = recvTuples.size();
        int chunkNums = 4*nthreads;
        if( nthreads*2 < recvTupleSize && recvTupleSize < chunkNums*2)
            chunkNums = nthreads;
        else if(recvTupleSize <= nthreads*2 )
            chunkNums = 1;
        vector<SpTuples<int64_t,double>*> ArrSpTups(chunkNums);
        int64_t chunkSize = std::floor(recvTupleSize / chunkNums);
        // double t1 = MPI_Wtime();

        #pragma omp parallel
        #pragma omp for schedule(dynamic)
        for (int i=0; i<chunkNums; i++)
        {
            int64_t myChunkSize = (i==chunkNums-1) ? (recvTupleSize - (chunkNums-1)*chunkSize) : chunkSize;
            ArrSpTups[i] = new SpTuples<int64_t, double>(myChunkSize, AspSeq->getnrow(), AspSeq->getncol(), &(recvTuples[chunkSize*i]), false, false);
        }
        // double t2 = MPI_Wtime();
        int64_t ArrSpTups_zize = 0; 
        for(int k=0; k<chunkNums; k++)
            ArrSpTups_zize += ArrSpTups[k]->getnnz();
        int64_t sortedtuple_size = ArrSpTups.size();    
        SpTuples<int64_t,double> * sortedTuples = MultiwayMerge<PTFF>(ArrSpTups, AspSeq->getnrow(), AspSeq->getncol(), false);
        // SpTuples<int64_t,double> * sortedTuples = MultiwayMergeHash<PTFF>(ArrSpTups,AspSeq->getnrow(), AspSeq->getncol());
        newspSeq = new Dist::DCCols(*sortedTuples, false);

        for(int i = 0; i < chunkNums; i++){
            ArrSpTups[i]->tuples_deleted = true; // Temporary patch to avoid memory leak and segfault
            delete ArrSpTups[i];
        }
        delete sortedTuples;
    }
    else
    {
        // newspSeq = new Dist::DCCols(static_cast<int64_t>(0), AspSeq->getnrow(), AspSeq->getncol(), static_cast<int64_t>(0));
        newspSeq = new Dist::DCCols(static_cast<int>(0), AspSeq->getnrow(), AspSeq->getncol(), static_cast<int>(0));
    }
    MPI_Barrier(World);
    Dist::MPI_DCCols newB(newspSeq, AcommGrid) ;
    
    MPI_Barrier(World);
    double loc_prune = MPI_Wtime();
    if(pruneA == true)
    {
        FullyDistSpVec<int, int64_t> sprvec(rvec, [](int64_t el){return el;});
        sprvec = sprvec.Invert(maxnrow);
        sprvec.Apply([](int64_t el){return 1;});
        FullyDistVec<int, int64_t> denseRi(sprvec);
        FullyDistSpVec<int, int64_t> spcvec(cvec, [](int64_t el){return el;});
        spcvec = spcvec.Invert(maxncol);
        spcvec.Apply([](int64_t el){return 1;});
        FullyDistVec<int, int64_t> denseCi(spcvec);
        Dist::MPI_DCCols Mask(A);
        Mask.DimApply(Row, denseRi, [](float mv, float vv){return vv;});
        Mask.PruneColumn(denseCi, [](float mv, float vv){return static_cast<float>((vv * mv)!=1);}, true);
        A.SetDifference(Mask);   
    }
    else if(pruneA == false)
    {
        A.SetDifference(newB);
    }

    MPI_Barrier(World);
    double local_add = MPI_Wtime();
    A += newB; 
    MPI_Barrier(World);
    double tend = MPI_Wtime();
    
    if(myrank == 0)
        printf("1st for count: %f, 2nd for threadaccum: %f 3rd for sendtuples: %f local newB:%f prune A: %f A+newB:%f\n", scnd_for_bgn-frst_for_bgn, third_for_bgn-scnd_for_bgn, forth_for_bgn-third_for_bgn, loc_prune-loc_newb, local_add-loc_prune, tend-local_add);

    double communication_time = comm_1_end - tstart + comm_2_end - comm_2_bgn + comm_3_end - comm_3_bgn + comm_4_end - comm_4_bgn + comm_5_end - comm_5_bgn;
    if(myrank == 0) printf( "my assign time is %f\n", tend - tstart );  
    if(myrank == 0) printf("my assign communication time:%f  \n End of Matrix Assignment \n\n", communication_time);
    fflush(stdout);
}