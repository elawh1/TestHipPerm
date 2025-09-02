// Permutation of sparse matrix corresponding to permutation vectors


#include <mpi.h>

// These macros should be defined before stdint.h is included
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif


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
#include "../CC.h"

using namespace std;
using namespace combblas;


class Dist
{
public:
    typedef SpDCCols < int64_t, double > DCCols;
    typedef SpParMat < int64_t, double, DCCols > MPI_DCCols;
};

int main(int argc, char* argv[])
{
    // -------------------- MPI init --------------------
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED)
    {
        printf("ERROR: The MPI library does not have MPI_THREAD_SERIALIZED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    int nthreads=1;
    
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
        // nthreads = 4;
    }
#endif
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank == 0)
    {
        cout << "Process Grid (p x p x t): " << sqrt(nprocs) << " x " << sqrt(nprocs) << " x " << nthreads << endl;
    }
    
    // -------------------- Parsing --------------------
    if(argc < 3)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./cc -I <mm|triples> -M <FILENAME_MATRIX_MARKET> (required)\n";
            cout << "-I <INPUT FILE TYPE> (mm: matrix market, triples: (vtx1, vtx2, edge_weight) triples. default:mm)\n";
            cout << "-base <BASE OF MATRIX MARKET> (default:1)\n";
            cout << "-rand <RANDOMLY PERMUTE VERTICES> (default:0)\n";
            cout << "Example (0-indexed mtx with random permutation): ./cc -M input.mtx -base 0 -rand 1" << endl;
            cout << "Example (triples format): ./cc -I triples -M input.txt" << endl;
        }
        MPI_Finalize();
        return -1;
    }
    {
        string ifilename = "", pvecfilename = "";
        int base = 0;
        int transpose = 0;
        bool isMatrixMarket = true;
        int savep = 0;
        int pvec = 0;
        
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i],"-I")==0)
            {
                string ifiletype = string(argv[i+1]);
                if(ifiletype == "triples") isMatrixMarket = false;
            }
            if (strcmp(argv[i],"-M")==0)
            {
                ifilename = string(argv[i+1]);
                if(myrank == 0) printf("filename: %s\n",ifilename.c_str());
            }
            else if (strcmp(argv[i],"-P")==0)
            {
                pvecfilename = string(argv[i+1]);
                if(myrank == 0) printf("permutation vec filename: %s\n",pvecfilename.c_str());
            }
            else if (strcmp(argv[i],"-base")==0)
            {
                base = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nBase of MM (1 or 0): %d\n",base);
            }
            else if (strcmp(argv[i],"-t")==0)
            {
                transpose = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nTranspose input matrix? (1 or 0): %d\n",transpose);
            }
            else if (strcmp(argv[i],"-savep")==0)
            {
                savep = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nSaving p matrix into a file? (1 or 0): %d\n",savep);
            }
            else if (strcmp(argv[i],"-pvec")==0)
            {
                pvec = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nReading permutation vector from file? (1 or 0): %d\n",pvec);
            }
            
        }

	    typedef PlusTimesSRing<double, double> PTFF;            //PTDOUBLEDOUBLE

        // ***** Reading File ******* //
        Dist::MPI_DCCols A(MPI_COMM_WORLD);	// construct object
        if(isMatrixMarket)
            A.ParallelReadMM(ifilename, base, maximum<double>());
        else
            A.ReadGeneralizedTuples(ifilename,  maximum<double>());
        
        SpParHelper::Print("input matrix read\n");

        if(transpose == 1)
        {
            A.Transpose();
            SpParHelper::Print("A transposed\n");
        }
       
        float loadImbalance = A.LoadImbalance();
        if (myrank == 0) printf("------- input matrix load imbalance: %f\n", loadImbalance);
        

        FullyDistVec<int64_t, int64_t> p ;
        if(A.getnrow() == A.getncol())
            p.iota(A.getnrow(), 0);
        else
            printf("input matrix is not square\n");
        fflush(stdout);
        p.RandPerm(); 
     
        FullyDistSpVec<int64_t, int64_t> pSp(p);
        pSp = pSp.Invert(A.getnrow());
        FullyDistVec<int64_t, int64_t> pidx(pSp);

        
        MPI_Comm World = p.commGrid->GetWorld();
       
   
        MPI_Comm ColWorld = p.commGrid->GetColWorld();
        MPI_Comm RowWorld = p.commGrid->GetRowWorld();
        auto commGrid = A.getcommgrid();
        int gridrows = commGrid->GetGridRows();
        int gridcols = commGrid->GetGridCols();

        double tstart, tend; 
        int rowneighs, rowrank, colrank, colneighs;
        MPI_Comm_size(ColWorld, &colneighs);
        MPI_Comm_size(RowWorld, &rowneighs);
        MPI_Comm_rank(RowWorld, &rowrank);
        MPI_Comm_rank(ColWorld, &colrank);
        int * localVecSize = new int[nprocs];  //changed rowsize tp localVecSize
        localVecSize[myrank] = p.LocArrSize();

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
        int64_t * localp = const_cast<int64_t*>(SpHelper::p2a(p.GetLocVec()));
        int * senddpls = new int[nprocs](); 
        int * sendcounts = new int[nprocs](); 
        for (int i=0; i<rowneighs; i++)
        {
            sendcounts[colrank+i*rowneighs] = localVecSize[myrank];
        }
        MPI_Barrier(World);
        double comm_2_bgn = MPI_Wtime();
        MPI_Alltoallv(localp, &sendcounts[0], senddpls, MPIType<int64_t>(), numacc_col, recvcount, dpls, MPIType<int64_t>(), World);
        MPI_Barrier(World);
        double comm_2_end = MPI_Wtime();
        delete [] dpls;
        delete [] recvcount;

        int * rowsize = new int[rowneighs];
        int * dpls_row = new int[rowneighs]();	// displacements (zero initialized pid) 
        for (int i=0; i<rowneighs; i++)
        {
            rowsize[i] = localVecSize[colrank*rowneighs+i];
        }
        std::partial_sum(rowsize, rowsize+rowneighs-1, dpls_row+1);
        int accsize = std::accumulate(rowsize, rowsize+rowneighs, 0);
        int64_t * numacc_row = new int64_t[accsize];
        int trxsize = p.LocArrSize();
        int64_t * trxnums = const_cast<int64_t*>(SpHelper::p2a(p.GetLocVec()));
        MPI_Barrier(World);
        double comm_3_bgn = MPI_Wtime();
        MPI_Allgatherv(trxnums, trxsize, MPIType<int64_t>(), numacc_row, rowsize, dpls_row, MPIType<int64_t>(), RowWorld);
        MPI_Barrier(World);
        double comm_3_end = MPI_Wtime();
        delete [] rowsize;
        delete [] dpls_row;

        MPI_Barrier(World);
        double pvec_end = MPI_Wtime();

        
        // ***** Start of matrix permutation ***** //
        int64_t m_perproc = A.getnrow() / gridrows;   //num of rows per processor
        int64_t n_perproc = A.getncol() / gridcols;
        Dist::DCCols *spSeq = A.seqptr();  
        Dcsc<int64_t, double>* Adcsc = spSeq->GetDCSC();
        int64_t nzc = 0;
        int64_t nz = 0;
        if(Adcsc != NULL && Adcsc != 0) 
        {
            nzc = Adcsc->nzc;
            nz = Adcsc->nz;
        }
        
        std::vector<int> thrdSegmentPntr(nthreads+1, 0);
        if(nz > 0 && nzc > 0)
        {
            int thrdSegNum = 0;
            thrdSegmentPntr[0] = 0;
            int64_t segmentSize = (nz)/nthreads;
            fflush(stdout);
            for (int64_t i=0; i<nzc-1; i++)
            {
                if (*((Adcsc->cp)+i) > (thrdSegNum+1)*(segmentSize))
                {
                    thrdSegNum++;
                    if(thrdSegNum < nthreads)
                        thrdSegmentPntr[thrdSegNum] = i;
                    else
                        thrdSegmentPntr[thrdSegNum] = nzc;
                }
            }
            for( int i=thrdSegNum+1; i<nthreads+1; i++)
                thrdSegmentPntr[i] = nzc;
        }
        
        int * sendcnt = new int[nprocs]();
        // std::vector<std::tuple<int64_t, int64_t, int64_t, double>> targetValues(nz);
        std::vector<int64_t> localSendAccum(nprocs);
        std::vector<vector<int64_t>> threadSendAccum(nthreads, localSendAccum);
        std::vector<int64_t> localSendCounter(nprocs);
        std::vector<std::vector<int64_t>> threadSendCounter(nthreads, localSendCounter);
        
        MPI_Barrier(World);
        double comp_1_bgn = MPI_Wtime();
        // omp_set_num_threads(nthreads);
        if (Adcsc != NULL && Adcsc!= 0)
        {
            #pragma omp parallel for schedule(static) num_threads(nthreads)
            for( int64_t k=0; k<nthreads; k++)
            {
                int myThread = omp_get_thread_num();
                for(int i=0; i<nprocs; i++)
                    threadSendCounter[myThread][i] = 0;
                for(int64_t i=thrdSegmentPntr[k]; i<thrdSegmentPntr[k+1]; i++)
                {
                    int newrowrank ;
                    int newcolrank ;
                    for(int64_t j = Adcsc->cp[i]; j< Adcsc->cp[i+1]; j++)
                    {
                        double tempval = Adcsc->numx[j] ;
                        int currlocalrow = Adcsc->ir[j];
                        int currlocalcol = Adcsc->jc[i];
                        if(n_perproc != 0)
                            newcolrank = std::min(static_cast<int>(numacc_col[currlocalcol] / n_perproc), gridcols-1);
                        else	// all owned by the last processor row
                            newcolrank = gridcols -1;
                        if(m_perproc != 0)
                            newrowrank = std::min(static_cast<int>(numacc_row[currlocalrow] / m_perproc), gridrows-1);
                        else	// all owned by the last processor row
                            newrowrank = gridrows -1;
                        int targetproc = commGrid->GetRank(newrowrank, newcolrank); 
                        int64_t newlcol = numacc_col[currlocalcol] - (newcolrank * n_perproc) ;
                        int64_t newlrow = numacc_row[currlocalrow] - (newrowrank * m_perproc) ;
                        // targetValues[j] = std::make_tuple(targetproc, newlrow, newlcol, tempval) ;
                        threadSendCounter[myThread][targetproc]++;
                    }
                }
            }
        }

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

        int * recvcnt = new int[nprocs]();
        MPI_Barrier(World);
        double comm_4_bgn = MPI_Wtime();   
        MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
        MPI_Barrier(World);
        double comm_4_end = MPI_Wtime();
        int * sdispls = new int[nprocs]();
        std::partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
        

        std::vector< std::tuple<int64_t,int64_t,double> > sendTuples(nz);
        MPI_Barrier(World);
        double comp_2_bgn = MPI_Wtime();
        if (Adcsc != NULL && Adcsc!= 0)
        {
            #pragma omp parallel for schedule(static) num_threads(nthreads)
            for( int64_t k=0; k<nthreads; k++)
            {
                int myThread = omp_get_thread_num();
                for(int i=0; i<nprocs; i++)
                    threadSendCounter[myThread][i] = 0;
                std::vector<int64_t> fillCounter(nprocs, static_cast<int64_t>(0));
                for(int64_t i=thrdSegmentPntr[k]; i<thrdSegmentPntr[k+1]; i++)
                {
                    int newrowrank ;
                    int newcolrank ;
                    for(int64_t j = Adcsc->cp[i]; j< Adcsc->cp[i+1]; j++)
                    {
                        double tempval = Adcsc->numx[j] ;
                        int currlocalrow = Adcsc->ir[j];
                        int currlocalcol = Adcsc->jc[i];
                        if(n_perproc != 0)
                            newcolrank = std::min(static_cast<int>(numacc_col[currlocalcol] / n_perproc), gridcols-1);
                        else	// all owned by the last processor row
                            newcolrank = gridcols -1;
                        if(m_perproc != 0)
                            newrowrank = std::min(static_cast<int>(numacc_row[currlocalrow] / m_perproc), gridrows-1);
                        else	// all owned by the last processor row
                            newrowrank = gridrows -1;
                        int targetproc = commGrid->GetRank(newrowrank, newcolrank); 
                        int64_t newlcol = numacc_col[currlocalcol] - (newcolrank * n_perproc) ;
                        int64_t newlrow = numacc_row[currlocalrow] - (newrowrank * m_perproc) ;
                        sendTuples[sdispls[targetproc]+threadSendAccum[k][targetproc]+fillCounter[targetproc]] = std::make_tuple(newlrow, newlcol, tempval);
                        fillCounter[targetproc]++;
                    }
                }
            }
        }
        double comp_2_end = MPI_Wtime(); 
        MPI_Barrier(World);

        MPI_Datatype MPI_tuple;
        MPI_Type_contiguous(sizeof(std::tuple<int64_t,int64_t,double>), MPI_CHAR, &MPI_tuple);
        MPI_Type_commit(&MPI_tuple);
        int64_t totrecv = accumulate(recvcnt, recvcnt+nprocs, static_cast<int64_t>(0));
        std::vector< std::tuple<int64_t,int64_t,double> > recvTuples(totrecv);
        int * rdispls = new int[nprocs]();

        std::partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);

        MPI_Barrier(World);
        double comm_5_bgn = MPI_Wtime();
        MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples.data(), recvcnt, rdispls, MPI_tuple, World);
        MPI_Barrier(World);
        double comm_5_end = MPI_Wtime();

        // ***** Change Local Matrix ***** //
        int64_t nrow = spSeq->getnrow() ;
        int64_t ncol = spSeq->getncol();
        Dist::DCCols* newspSeq;
        MPI_Barrier(World);
        double loc_newb = MPI_Wtime();
        if(recvTuples.size() > 0)
        {   
            int recvTupleSize = recvTuples.size();
            int chunkNums = nthreads;
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
                integerSort(&(recvTuples[chunkSize*i]), myChunkSize);
                
                ArrSpTups[i] = new SpTuples<int64_t, double>(myChunkSize, nrow, ncol, &(recvTuples[chunkSize*i]), true, false);
            }
            SpTuples<int64_t,double> * sortedTuples = MultiwayMergeHash<PTFF>(ArrSpTups, nrow, ncol);
              
            newspSeq = new Dist::DCCols(*sortedTuples, false);
            for(int i = 0; i < chunkNums; i++){
                ArrSpTups[i]->tuples_deleted = true; // Temporary patch to avoid memory leak and segfault
                delete ArrSpTups[i];    
            }
        }
        else
        {
            newspSeq = new Dist::DCCols(static_cast<int64_t>(0), nrow, ncol, static_cast<int64_t>(0));
        }
        Dist::MPI_DCCols V(newspSeq, commGrid) ;
      
        MPI_Barrier(World);
        tend = MPI_Wtime();
        double communication_time = comm_1_end - tstart + comm_2_end - comm_2_bgn + comm_3_end - comm_3_bgn + comm_4_end - comm_4_bgn + comm_5_end - comm_5_bgn;
    
        if(myrank == 0)
        {
            printf( "my permutation time is %f\n", tend - tstart );  
            fflush(stdout);
            double communication_time = comm_1_end - tstart + comm_2_end - comm_2_bgn + comm_3_end - comm_3_bgn + comm_4_end - comm_4_bgn + comm_5_end - comm_5_bgn;
            double total_time = tend - tstart;
            printf( "my permutation communication time is %f\n", communication_time );  
            fflush(stdout);
            fflush(stdout);
            printf("local computation multi threaded : %f", comm_4_bgn-pvec_end+comp_2_end-comp_2_bgn);
            printf("First Part: p vector gathering : %f\n", pvec_end-tstart);
            printf("Second Part: local computation : %f\n", comm_5_bgn-pvec_end-(comm_4_end - comm_4_bgn));
            printf("Third part: exchange data %f\n", comm_5_end - comm_5_bgn );  
            printf("Forth(Last) Part: build local matrix : %f\n", tend-comm_5_end);     
            fflush(stdout);
        }
        
        
        
        double ttstart, ttend;
        MPI_Barrier(World);
        ttstart = MPI_Wtime();
        A(pidx, pidx, true);
        MPI_Barrier(World);
        ttend = MPI_Wtime();
        if (myrank == 0)
            printf( "spref perm  time is %f\n", ttend - ttstart );  
        fflush(stdout);

        float AloadImbalance = A.LoadImbalance();
        if (myrank == 0) printf("------- A load imbalance after our permutation: %f\n", AloadImbalance);
        float VloadImbalance = V.LoadImbalance();
        if (myrank == 0) printf("------- A load imbalance after spref permutation: %f\n", VloadImbalance);

        
        A.PrintInfo();
        V.PrintInfo();
        if( V == A )
            SpParHelper::Print("\n A==V \n");
        else
            SpParHelper::Print("\n permutation results isn't same\n");
        fflush(stdout);
        
	}
    
    MPI_Finalize();
    return 0; 

}