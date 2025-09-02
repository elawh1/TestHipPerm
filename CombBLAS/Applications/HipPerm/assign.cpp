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
#include "../CC.h"

#include <unordered_set>

using namespace std;
using namespace combblas;

class Dist
{
public:
    typedef SpDCCols<int64_t, double> DCCols;
    typedef SpParMat<int64_t, double, DCCols> MPI_DCCols;
};

// Function to perform matrix assignment, receiving matrices and vectors as parameters
void MatrixAssignment(Dist::MPI_DCCols &A, Dist::MPI_DCCols &B, FullyDistVec<int64_t, int64_t> &rvec, FullyDistVec<int64_t, int64_t> &cvec, bool pruneA, MPI_Comm World) {
    
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
    
    int accsize_col = std::accumulate(recvcount, recvcount+nprocs, 0);
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
    int * dpls_row = new int[rowneighs]();	// displacements (zero initialized pid) 

    // #pragma omp parallel for
    for (int i=0; i<rowneighs; i++)
        rowsize[i] = localVecSize[colrank*rowneighs+i];
    
    std::partial_sum(rowsize, rowsize+rowneighs-1, dpls_row+1);        
    int accsize = std::accumulate(rowsize, rowsize+rowneighs, 0);
    int64_t * numacc_row = new int64_t[accsize];
    int trxsize = rvec.LocArrSize();
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
                    // double tempval = * ((Bdcsc->numx)+j) ;
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
    MPI_Type_contiguous(sizeof(std::tuple<int64_t,int64_t,double>), MPI_CHAR, &MPI_tuple);
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
    std::vector< std::tuple<int64_t,int64_t,double> > sendTuples(totsend);
    if (Bdcsc != NULL && Bdcsc!= 0)
    {
        #pragma omp parallel for schedule(static) num_threads(nthreads)
        for( int64_t k=0; k<nthreads; k++)
        {
            int newrowrank ;
            int newcolrank ;
            std::vector<int64_t> fillCounter(nprocs, static_cast<int64_t>(0));
            int myThread = omp_get_thread_num();
            for(int64_t i=thrdSegmentPntr[k]; i<thrdSegmentPntr[k+1]; i++)
            {
                for(int64_t j = *((Bdcsc->cp)+i); j< *((Bdcsc->cp)+i+1); j++)
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
                    int64_t newlcol = numacc_col[currlocalcol] - (newcolrank * out_n_perproc) ;
                    int64_t newlrow = numacc_row[currlocalrow] - (newrowrank * out_m_perproc) ;
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

    int64_t Anrow = A.getnrow();
    int64_t Ancol = A.getncol();
    
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
        #pragma omp parallel
        #pragma omp for schedule(dynamic)
        for (int i=0; i<chunkNums; i++)
        {
            int64_t myChunkSize = (i==chunkNums-1) ? (recvTupleSize - (chunkNums-1)*chunkSize) : chunkSize;
            ArrSpTups[i] = new SpTuples<int64_t, double>(myChunkSize, AspSeq->getnrow(), AspSeq->getncol(), &(recvTuples[chunkSize*i]), false, false);
        }
        SpTuples<int64_t,double> * sortedTuples = MultiwayMerge<PTFF>(ArrSpTups, AspSeq->getnrow(), AspSeq->getncol(), false);
        newspSeq = new Dist::DCCols(*sortedTuples, false);
        for(int i = 0; i < chunkNums; i++){
            ArrSpTups[i]->tuples_deleted = true; // Temporary patch to avoid memory leak and segfault
            delete ArrSpTups[i];
        }
    }
    else
    {
        newspSeq = new Dist::DCCols(static_cast<int64_t>(0), AspSeq->getnrow(), AspSeq->getncol(), static_cast<int64_t>(0));
    }
    
    Dist::MPI_DCCols newB(newspSeq, AcommGrid) ;

    MPI_Barrier(World);
    double loc_prune = MPI_Wtime();
    if(pruneA == true)
    {
        FullyDistSpVec<int64_t, int64_t> sprvec(rvec, [](int64_t el){return el;});
        sprvec = sprvec.Invert(Anrow);
        sprvec.Apply([](int64_t el){return 1;});
        FullyDistVec<int64_t, int64_t> denseRi(sprvec);
        FullyDistSpVec<int64_t, int64_t> spcvec(cvec, [](int64_t el){return el;});
        spcvec = spcvec.Invert(Ancol);
        spcvec.Apply([](int64_t el){return 1;});
        FullyDistVec<int64_t, int64_t> denseCi(spcvec);
        Dist::MPI_DCCols Mask(A);
        Mask.DimApply(Row, denseRi, [](float mv, float vv){return vv;});
        Mask.PruneColumn(denseCi, [](float mv, float vv){return static_cast<float>((vv * mv)!=1);}, true);
        A.SetDifference(Mask);   
        ///// ------ End of Prune ----- //     
    }
    else if(pruneA == false)
    {
        // we just delete existing edges in intersection of rvec and cvec
        A.SetDifference(newB);
    }

    MPI_Barrier(World);
    double local_add = MPI_Wtime();
    A += newB; 
    MPI_Barrier(World);
    tend = MPI_Wtime();
 
    if(myrank == 0)
        printf("1st for count: %f, 2nd for threadaccum: %f 3rd for sendtuples: %f local newB:%f prune A: %f A+newB:%f\n", scnd_for_bgn-frst_for_bgn, third_for_bgn-scnd_for_bgn, forth_for_bgn-third_for_bgn, loc_prune-loc_newb, local_add-loc_prune, tend-local_add);


    double communication_time = comm_1_end - tstart + comm_2_end - comm_2_bgn + comm_3_end - comm_3_bgn + comm_4_end - comm_4_bgn + comm_5_end - comm_5_bgn;
    if(myrank == 0) printf( "my assign time is %f\n", tend - tstart );  
    if(myrank == 0) printf("my assign communication time:%f\n", communication_time);
    fflush(stdout);

}

int main(int argc, char* argv[])
{

    int nthreads = 1;
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED)
    {
        printf("ERROR: The MPI library does not have MPI_THREAD_SERIALIZED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm World = MPI_COMM_WORLD;

    int nprocs, myrank;
    MPI_Comm_size(World,&nprocs);
    MPI_Comm_rank(World, &myrank);

    if(myrank == 0)
    {
        cout << "Process Grid (p x p x t): " << sqrt(nprocs) << " x " << sqrt(nprocs) << " x " << nthreads << endl;
    }

    if(argc < 3)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./stream_experiment_all -I <mm|triples> -path <Path to Files> (required) -base 1\n"<< endl;
            cout << "-I <INPUT FILE TYPE> (mm: matrix market, triples: (vtx1, vtx2, edge_weight) triples. default:mm)\n"<< endl;
            cout << "-base <BASE OF MATRIX MARKET> (default:1)\n"<< endl;
        }
        MPI_Finalize();
        return -1;
    }
    {
        // Argument parsing and initial setup
        string ifilename = "", ibfilename = "", rvec_filename = "", cvec_filename = "";
        string vec1_filename = "";
        string vec2_filename = "";
        string filePath = "";
        int base = 0;
        bool pruneA = false;

        // Parse command-line arguments
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "-A") == 0) {
                ifilename = string(argv[i + 1]);
                if (myrank == 0) printf("filename: %s\n", ifilename.c_str());
            }
            if (strcmp(argv[i], "-B") == 0) {
                ibfilename = string(argv[i + 1]);
                if (myrank == 0) printf("filenameB: %s\n", ibfilename.c_str());
            }
            else if (strcmp(argv[i], "-base") == 0) {
                base = atoi(argv[i + 1]);
                if (myrank == 0) printf("\nBase of MM (1 or 0): %d\n", base);
            }
            if (strcmp(argv[i], "-d1") == 0) {
                rvec_filename = string(argv[i + 1]);
                if (myrank == 0) printf("vec1 filename: %s\n", rvec_filename.c_str());
            }
            if (strcmp(argv[i], "-d2") == 0) {
                cvec_filename = string(argv[i + 1]);
                if (myrank == 0) printf("vec2 filename: %s\n", cvec_filename.c_str());
            }
            if (strcmp(argv[i],"-pruneA")==0)
            {
                int prune = atoi(argv[i+1]);
                if(prune == 1) pruneA = true;
                if(myrank == 0) printf("make holes in matrix A: %d\n", pruneA);
            }
        }
    
        // Define combBLAS matrix objects
        Dist::MPI_DCCols A(World), B(World);

        // Read matrices from files
        // ifilename = filePath + "Aext_res.mtx";
        A.ParallelReadMM(ifilename, base, maximum<double>());
        SpParHelper::Print("reading file A done\n");

        // ibfilename = filePath + "A_seg1.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        SpParHelper::Print("reading file B done\n");

        FullyDistVec<int64_t, int64_t> vec1, vec2;
        vec1.ParallelRead(rvec_filename, 0, maximum<int64_t>());
        vec2.ParallelRead(cvec_filename, 0, maximum<int64_t>());
        SpParHelper::Print("reading vector segments done\n");

        // Call the matrix assignment function with the read data
        MPI_Barrier(World);
        double tstasgn2 = MPI_Wtime();
        MatrixAssignment(A, B, vec1, vec2, pruneA, World);
        MPI_Barrier(World);
        double tendasgn2 = MPI_Wtime();

        SpParHelper::Print("Returned from matrixAssignement\n");
        A.PrintInfo();

        ostringstream  outs;
        outs << "Total asgn time: " << tendasgn2-tstasgn2 << endl;
        SpParHelper::Print(outs.str());
    }
    
    MPI_Finalize();
    return 0;
}
