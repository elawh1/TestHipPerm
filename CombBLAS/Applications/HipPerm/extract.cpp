// This is for testing random permutation method of combBLAS

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
    int provided;
    double max_time = 0.0, max_comm_time = 0.0 ; 
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
    }
#endif
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(myrank == 0)
    {
        cout << "Process Grid (p x p x t): " << sqrt(nprocs) << " x " << sqrt(nprocs) << " x " << nthreads << endl;
    }
    
    if(argc < 3)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./cc -I <mm|triples> -M <FILENAME_MATRIX_MARKET> (required)\n";
            cout << "-I <INPUT FILE TYPE> (mm: matrix market, triples: (vtx1, vtx2, edge_weight) triples. default:mm)\n";
            cout << "-base <BASE OF MATRIX MARKET> (default:1)\n";
            cout << "Example (0-indexed mtx with random permutation): ./cc -M input.mtx -base 0 -rand 1" << endl;
            cout << "Example (triples format): ./cc -I triples -M input.txt" << endl;
        }
        MPI_Finalize();
        return -1;
    }
    {
        string ifilename = "";
        string rvec_filename = "";
        string cvec_filename = "";
        int64_t Bnrow, Bncol;
        int base = 0;
        int vecbase = 0;
        bool isMatrixMarket = true;
        
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
                if(myrank == 0) printf("matrix filename: %s\n",ifilename.c_str());
            }
            if (strcmp(argv[i],"-d1")==0)
            {
                rvec_filename = string(argv[i+1]);
                if(myrank == 0) printf("vec1 filename: %s\n",rvec_filename.c_str());
            }
            if (strcmp(argv[i],"-d2")==0)
            {
                cvec_filename = string(argv[i+1]);
                if(myrank == 0) printf("vec2 filename: %s\n",cvec_filename.c_str());
            }
            else if (strcmp(argv[i],"-base")==0)
            {
                base = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nBase of MM (1 or 0):%d",base);
            }
            else if (strcmp(argv[i],"-vecbase")==0)
            {
                vecbase = atoi(argv[i + 1]);
                if(myrank == 0) printf("\nBase of rvec cvec (1 or 0):%d", vecbase);
            }
        }

	    typedef PlusTimesSRing<double, double> PTFF;            //PTDOUBLEDOUBLE

        // ***** Reading File ******* //
        Dist::MPI_DCCols A(MPI_COMM_WORLD);	// construct object
        if(isMatrixMarket)
            A.ParallelReadMM(ifilename, base, maximum<double>());
        else
            A.ReadGeneralizedTuples(ifilename,  maximum<double>());
        SpParHelper::Print("reading A matrix done\n");
        
        auto commGrid = A.getcommgrid();
        int gridrows = commGrid->GetGridRows();
        int gridcols = commGrid->GetGridCols();
        MPI_Comm World = commGrid->GetWorld();
        MPI_Comm ColWorld = commGrid->GetColWorld();
        MPI_Comm RowWorld = commGrid->GetRowWorld();
        int rowneighs, rowrank, colrank;
        MPI_Comm_size(RowWorld, &rowneighs);
        MPI_Comm_rank(RowWorld, &rowrank);
        MPI_Comm_rank(ColWorld, &colrank);
        int64_t Anrow = A.getnrow();
        int64_t Ancol = A.getncol();
        int64_t m_perproc = Anrow / gridrows;   //num of rows per processor except last in col
        int64_t n_perproc = Ancol / gridcols;   //num of cols per processor except last in row

        // ************ read rvec and cvec  from file ************ ///
        FullyDistVec<int64_t, int64_t> rvec, cvec ;
        rvec.ParallelRead(rvec_filename, vecbase, maximum<int64_t>());
        cvec.ParallelRead(cvec_filename, vecbase, maximum<int64_t>());
        SpParHelper::Print("reading rvec and cvec done\n");
        fflush(stdout);


        Bnrow = rvec.glen;
        Bncol = cvec.glen;

        int64_t max_rvec = rvec.Reduce(maximum<int64_t>(), 0.0);
        int64_t max_cvec = cvec.Reduce(maximum<int64_t>(), 0.0);
        if(myrank ==0)  printf("A size: %d * %d ... B size: %d * %d ... max rvec: %d ... max cvec: %d\n", Anrow, Ancol, Bnrow, Bncol, max_rvec, max_cvec);
        fflush(stdout);

        int64_t Bm_perproc = Bnrow /gridrows;
        int64_t Bn_perproc = Bncol /gridcols;

        double tstart, tend; 
        MPI_Barrier(World);
        tstart = MPI_Wtime();

        // ***** generating spvec from dense vec **** //
        MPI_Barrier(World);
        double sp_bgn = MPI_Wtime();
        FullyDistSpVec<int64_t, int64_t> sprvec(rvec);
        FullyDistSpVec<int64_t, int64_t> spcvec(cvec);
        MPI_Barrier(World);
        double sp_end = MPI_Wtime();

        MPI_Barrier(World);
        double invert_bgn = MPI_Wtime();
        sprvec = sprvec.Invert(Anrow);
        spcvec = spcvec.Invert(Ancol);
        MPI_Barrier(World);
        double invert_end = MPI_Wtime();

        // ***** cvec gathering **** //
        int * localcVecSize = new int[nprocs];  //changed rowsize tp localcVecSize
        localcVecSize[myrank] = spcvec.getlocnnz();

        MPI_Barrier(World);
        double comm_6_bgn = MPI_Wtime();
        MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, localcVecSize, 1, MPI_INT, World);
        MPI_Barrier(World);
        double comm_6_end = MPI_Wtime();

        int * dpls = new int[nprocs]();	// displacements (zero initialized pid) 
        int * myColVecSize = new int[nprocs]();
        for (int i=rowrank*rowneighs; i<nprocs; i++)
        {
            if (i < rowrank*rowneighs+rowneighs)
                myColVecSize[i] = localcVecSize[i];
            if((i > rowrank*rowneighs) && (i < rowrank*rowneighs+rowneighs))
                dpls[i] = myColVecSize[i-1] +dpls[i-1];
            else if(i >= rowrank*rowneighs+rowneighs)
                dpls[i] = dpls[i-1];
        }
       
        int accsize_col = std::accumulate(myColVecSize, myColVecSize+nprocs, 0);
        std::vector<int64_t> myColVec(accsize_col);
        std::vector<int64_t> myColVecInd(accsize_col);
      
        int vecOffset;
        if(colrank < gridcols-1)  
            vecOffset = rowrank * (int)(n_perproc / gridcols);
        else
        {
            int64_t nonRemaining  = n_perproc*gridcols-n_perproc;
            int64_t remainingcols = (Ancol - nonRemaining);
            int divis = remainingcols / gridcols;
            vecOffset = rowrank * divis;
        }
        MPI_Barrier(World);
        double lsp_bgn = MPI_Wtime();
        std::vector<int64_t> localIndices = spcvec.GetLocalInd();
        std::vector<int64_t> localValues = spcvec.GetLocalNum();
        #pragma omp parallel for
        for( int i=0; i<spcvec.getlocnnz(); i++)
        {
            localIndices[i] += vecOffset;
        }
        MPI_Barrier(World);
        double lsp_end = MPI_Wtime();
        int * senddpls = new int[nprocs](); 
        int * sendcounts = new int[nprocs](); 
        for (int i=0; i<rowneighs; i++)
        {
            sendcounts[colrank+i*rowneighs] = localcVecSize[myrank];
        }
        
        fflush(stdout);
        MPI_Barrier(World);
        double comm_2_bgn = MPI_Wtime();
        //send indices first
        MPI_Alltoallv(localIndices.data(), &sendcounts[0], senddpls,  MPI_INT64_T, myColVecInd.data(), myColVecSize, dpls, MPI_INT64_T, World);
        MPI_Alltoallv(localValues.data(), &sendcounts[0], senddpls,  MPI_INT64_T, myColVec.data(), myColVecSize, dpls, MPI_INT64_T, World);
        MPI_Barrier(World);
        double comm_2_end = MPI_Wtime();

        localIndices.clear();
        localValues.clear();
        delete[] localcVecSize; 
        delete[] dpls; 
        delete[] myColVecSize;
        delete[] senddpls;

        // ***** rvec gathering **** //
        int * localrVecSize = new int[rowneighs];
        localrVecSize[rowrank] = sprvec.getlocnnz();
        MPI_Barrier(World);
        double comm_1_bgn = MPI_Wtime();
        MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, localrVecSize, 1, MPI_INT, RowWorld);
        MPI_Barrier(World);
        double comm_1_end = MPI_Wtime();
        int * dpls_row = new int[rowneighs]();	// displacements (zero initialized pid) 
        std::partial_sum(localrVecSize, localrVecSize+rowneighs-1, dpls_row+1);
        int accsize = std::accumulate(localrVecSize, localrVecSize+rowneighs, 0);
        int trxsize = sprvec.getlocnnz();

        std::vector<int64_t> myRowVec(accsize);
        std::vector<int64_t> myRowVecInd(accsize);
        
        std::vector<int64_t> rveclocalIndices = sprvec.GetLocalInd();
        std::vector<int64_t> rveclocalValues = sprvec.GetLocalNum();
        #pragma omp parallel for
        for( int i=0; i<sprvec.getlocnnz(); i++)
        {
            rveclocalIndices[i] += vecOffset;
        }

        MPI_Barrier(World);
        double comm_3_bgn = MPI_Wtime();
        MPI_Allgatherv(rveclocalValues.data(), trxsize, MPI_INT64_T, myRowVec.data(), localrVecSize, dpls_row, MPI_INT64_T, RowWorld);
        MPI_Allgatherv(rveclocalIndices.data(), trxsize, MPI_INT64_T, myRowVecInd.data(), localrVecSize, dpls_row, MPI_INT64_T, RowWorld);
        MPI_Barrier(World);
        double comm_3_end = MPI_Wtime();
        delete [] localrVecSize;
        delete [] dpls_row;

        MPI_Barrier(World);
        double pvec_end = MPI_Wtime();
      

        // ***** Start of matrix extraction ***** //
        Dist::DCCols *spSeq = A.seqptr();  
        Dcsc<int64_t, double>* Adcsc = spSeq->GetDCSC();
        int64_t nzc=0 , nz=0;
        if (Adcsc != NULL && Adcsc!= 0)
        {
            nzc = Adcsc->nzc;
            nz = Adcsc->nz;
        }

        MPI_Barrier(World);
        double ttt = MPI_Wtime();
        std::vector<int64_t> coleqidx(nthreads+1, nzc);
        coleqidx[0] = 0;
        int jstrt = 0;
        if(accsize_col != 0)
        {
            for( int64_t p=1; p<nthreads ; p++)
            {
                int ibgn = floor(p * (static_cast<double>(accsize_col) / nthreads));  
                // printf("----accsize_col: %d  nthreads: %d  p: %d  ibg: %d\n", accsize_col, nthreads, p, ibgn);
                int64_t currCol = myColVecInd[ibgn];
                int64_t* jcStart = Adcsc->jc + jstrt; 
                int64_t* jcEnd = Adcsc->jc + nzc;     
                int64_t* it = std::lower_bound(jcStart, jcEnd, currCol); 
                coleqidx[p] = it - Adcsc->jc; 
                jstrt = coleqidx[p];  
            }   
        } 
        coleqidx[nthreads] = nzc;
        MPI_Barrier(World);
        double ttte = MPI_Wtime();

        int64_t targetSize = std::floor(nz/nthreads)>0 ? std::floor(nz/nthreads) : nz;
        std::vector<std::tuple<int64_t, int64_t, int64_t, double>> targetValues(targetSize);
        std::vector< std::vector< std::tuple<int64_t, int64_t, int64_t, double>>> allTargetValues(nthreads, targetValues);
        std::vector<int64_t> targetValCnt(nthreads, 0);
        
        std::vector< std::vector <int64_t>> thrdSendCnt( nthreads, std::vector<int64_t>(nprocs, static_cast<int64_t>(0)));
        double for_bgn = MPI_Wtime();
        #pragma omp parallel 
        if (Adcsc != NULL && Adcsc!=0 && accsize_col!=0 && accsize!=0)
        {
            
            #pragma omp for schedule(static)
            for( int64_t p=0; p<nthreads; p++)
            {
                int threadID = p;
                int ibgn = floor(threadID*( static_cast<double>(accsize_col)/nthreads));
                int iend = (threadID == nthreads-1) ? accsize_col-1 : floor((threadID+1)*(static_cast<double>(accsize_col)/nthreads))-1;
                int64_t newcolrank, newrowrank;
                bool jfound = false;
                bool ifound = false;
                for(int i= coleqidx[threadID]; i <= coleqidx[threadID+1]-1; i++)
                {
                    int64_t currCol = Adcsc->jc[i];
                    iend = (threadID == nthreads-1) ? accsize_col-1 : floor((threadID+1)*(static_cast<double>(accsize_col)/nthreads))-1;
                    ibgn = floor(threadID*(static_cast<double>(accsize_col)/nthreads));
                    jfound = false;
                    while (ibgn <= iend &&  jfound == false) { 
                        int64_t j = (iend + ibgn) / 2;
                         
                        if (myColVecInd[j] == currCol)
                        {
                        
                            int64_t irPtr = Adcsc->cp[i];
                            int64_t irPtrEnd = Adcsc->cp[i+1];
                            int64_t kbgn = 0;
                            int64_t kend = accsize-1;

                            for( int l=irPtr; l<irPtrEnd; l++)
                            {  
                                int64_t currRow = Adcsc->ir[l];
                                kend = accsize-1;
                                ifound = false;
                                kbgn = 0;
                                while (kbgn <= kend && ifound == false) {
                                    int64_t k = (kbgn + kend)  / 2;

                                    
                                    if (myRowVecInd[k] == currRow)
                                    {
                                       double tempval = Adcsc->numx[l] ;
                                        if(Bn_perproc != 0)
                                            newcolrank = std::min(static_cast<int>(myColVec[j] / Bn_perproc), gridcols-1);
                                        else	// all owned by the last processor row
                                            newcolrank = gridcols -1;
                                        if(Bm_perproc != 0)
                                            newrowrank = std::min(static_cast<int>(myRowVec[k] / Bm_perproc), gridrows-1);
                                        else	// all owned by the last processor row
                                            newrowrank = gridrows -1;

                                        int targetproc = commGrid->GetRank(newrowrank, newcolrank); 
                                        int64_t newlcol = myColVec[j] - (newcolrank * Bn_perproc) ;
                                        int64_t newlrow = myRowVec[k] - (newrowrank * Bm_perproc) ;
                                        thrdSendCnt[threadID][targetproc]++;
                                        if(targetValCnt[threadID] >= allTargetValues[threadID].size())
                                            allTargetValues[threadID].resize(allTargetValues[threadID].size()+targetSize);
                                        allTargetValues[threadID][targetValCnt[threadID]] = std::make_tuple(targetproc, newlrow, newlcol, tempval) ;
                                        targetValCnt[threadID]++;

                                        ifound = true;
                                    }
                                    else if(myRowVecInd[k] < currRow)
                                        kbgn = k + 1 ;
                                    else
                                        kend = k - 1;
                                }
                            }
                            jfound = true;
                        }  
                        else if (myColVecInd[j] < currCol)
                            ibgn = j + 1;
                        else
                            iend = j - 1; 
                    }
                }
            }
        }

        MPI_Barrier(World);
        std::vector<int64_t>().swap(myRowVecInd);
        std::vector<int64_t>().swap(myRowVec);
        double for_end = MPI_Wtime();

        std::vector<int64_t> localSendAccum(nprocs, 0);
        std::vector<vector<int64_t>> threadSendAccum(nthreads, localSendAccum);
        int * sendcnt = new int[nprocs]();
        #pragma omp parallel for 
        for( int j=0 ; j<nprocs; j++)
        {
            threadSendAccum[0][j] = 0;
            sendcnt[j] += thrdSendCnt[0][j];
            for( int i=1; i<nthreads; i++)
            {
                threadSendAccum[i][j] = thrdSendCnt[i-1][j] + threadSendAccum[i-1][j];
                sendcnt[j] += thrdSendCnt[i][j];
            }

        }  


        int * recvcnt = new int[nprocs]();
        MPI_Barrier(World);
        double comm_4_bgn = MPI_Wtime();   
        MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
        MPI_Barrier(World);
        double comm_4_end = MPI_Wtime();
        
        int * sdispls = new int[nprocs]();
        int * rdispls = new int[nprocs]();
        std::partial_sum(sendcnt, sendcnt+nprocs-1, sdispls+1);
        std::partial_sum(recvcnt, recvcnt+nprocs-1, rdispls+1);
        int64_t totrecv = accumulate(recvcnt, recvcnt+nprocs, static_cast<int64_t>(0));
        int64_t totsend = accumulate(sendcnt, sendcnt+nprocs, static_cast<int64_t>(0));
        std::vector< std::tuple<int64_t,int64_t,double> > sendTuples(totsend);



        MPI_Barrier(World);
        double thread_3_bgn = MPI_Wtime();

        #pragma omp parallel
        if(totsend != 0)
        {

            std::vector<int64_t> fillCounter(nprocs, static_cast<int64_t>(0));
            #pragma omp for schedule(static) 
            for( int64_t p=0; p<nthreads; p++)
            {
                int threadID = p;
                for(int i=0; i<targetValCnt[threadID]; i++)
                {
                    auto tuple = allTargetValues[threadID][i];
                    int targetproc = std::get<0>(tuple);
                    sendTuples[sdispls[targetproc]+fillCounter[targetproc]+threadSendAccum[threadID][targetproc]] = std::make_tuple(std::get<1>(tuple), std::get<2>(tuple), std::get<3>(tuple));
                    fillCounter[targetproc]++;
                }
            }
        }
        allTargetValues.clear();
  
        MPI_Barrier(World);
        double thread_3_end = MPI_Wtime();
        
        MPI_Datatype MPI_tuple;
        MPI_Type_contiguous(sizeof(std::tuple<int64_t,int64_t,double>), MPI_CHAR, &MPI_tuple);
        MPI_Type_commit(&MPI_tuple);
        std::vector< std::tuple<int64_t,int64_t,double> > recvTuples(totrecv);

        MPI_Barrier(World);
        double comm_5_bgn = MPI_Wtime();
        MPI_Alltoallv(sendTuples.data(), sendcnt, sdispls, MPI_tuple, recvTuples.data(), recvcnt, rdispls, MPI_tuple, World);
        MPI_Barrier(World);
        double comm_5_end = MPI_Wtime();
        DeleteAll(sendcnt, recvcnt, sdispls, rdispls); // free all memory
        std::vector<std::tuple<int64_t, int64_t, double>>().swap(sendTuples);  // free all memory
        MPI_Type_free(&MPI_tuple);
 
        // ***** Change Local Matrix ***** //
        int64_t nrow;
        int64_t ncol;
        if (rowrank < gridrows-1)
            ncol = Bn_perproc;
        else
            ncol = Bncol - (gridrows-1)*Bn_perproc ;
        if (colrank < gridcols-1)
            nrow = Bm_perproc;
        else
            nrow = Bnrow - (gridcols-1)*Bm_perproc;
        

        Dist::DCCols* newspSeq;
        MPI_Barrier(World);
        double loc_newb = MPI_Wtime();
        // int chunkNums = 4*nthreads;
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
                ArrSpTups[i] = new SpTuples<int64_t, double>(myChunkSize, nrow, ncol, &(recvTuples[chunkSize*i]), false, false);
            }
            SpTuples<int64_t,double> * sortedTuples = MultiwayMerge<PTFF>(ArrSpTups, nrow, ncol, false);
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

        //Free 
        std::vector<std::tuple<int64_t, int64_t, double>>().swap(recvTuples);
        

        Dist::MPI_DCCols V(newspSeq, commGrid) ;

        // delete newspSeq;
        MPI_Barrier(World);
        tend = MPI_Wtime();
        if(myrank == 0)
        {
            printf( "my extract time is %f\n", tend - tstart );  
            printf("rvec cvec gathering:%f\n", pvec_end - tstart);
            printf("local comp before alltoall:%f\n", thread_3_end-pvec_end);
            printf("exchange data:%f\n", comm_5_end-thread_3_end);
            printf("build local matrix : %f\n", tend-comm_5_end);
            
        }
        fflush(stdout);
        if(myrank == 0)
        {
            double communication_time = comm_1_end - comm_1_bgn + comm_2_end - comm_2_bgn + comm_3_end - comm_3_bgn + comm_4_end - comm_4_bgn + comm_5_end - comm_5_bgn + comm_6_end - comm_6_bgn;
            printf( "my extract communication time is %f\n", communication_time );  
            fflush(stdout);
        }
        
        double ttstart, ttend;
        Dist::MPI_DCCols B(MPI_COMM_WORLD);
        MPI_Barrier(World);
        ttstart = MPI_Wtime();
        B = A(rvec, cvec, false);
        MPI_Barrier(World);
        ttend = MPI_Wtime();
        if (myrank == 0)
            printf( "combBLAS extract time is %f\n", ttend - ttstart );  
        fflush(stdout);

        SpParHelper::Print("My extracted:\n");
        V.PrintInfo();
        SpParHelper::Print("CombBLAS extracted:\n");
        B.PrintInfo();
        fflush(stdout);
        if( V == B )
            SpParHelper::Print("A==V \n");
        else
            SpParHelper::Print("extract result isn't same\n");
        fflush(stdout);
    }
    
    MPI_Finalize();
    return 0; 

}