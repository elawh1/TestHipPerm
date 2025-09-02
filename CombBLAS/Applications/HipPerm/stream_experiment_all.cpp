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
#include "stream_experiment_all.h"

#include <unordered_set>

using namespace std;
using namespace combblas;

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
        string ifilename = "", ibfilename = "";
        string vec1_filename = "";
        string vec2_filename = "";
        string vec3_filename = "";
        string filePath = "";
        int base = 0;
        bool pruneA = false;

        // Parse command-line arguments
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "-path") == 0) {
                filePath = string(argv[i + 1]);
                // if (myrank == 0) printf("filename: %s\n", i  filename.c_str());
            }
            else if (strcmp(argv[i], "-base") == 0) {
                base = atoi(argv[i + 1]);
                if (myrank == 0) printf("\nBase of MM (1 or 0): %d\n", base);
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
        float B_load, A_load, Anew_load;
        double tstasgn1, tendasgn1, tstasgn2, tendasgn2, tstasgn3, tendasgn3, tstasgn4, tendasgn4, tstasgn5, tendasgn5, tstasgn6, tendasgn6, tstasgn7, tendasgn7, tstasgn8, tendasgn8, tstasgn9, tendasgn9;
        double tmidasgn1, tmidasgn2, tmidasgn3, tmidasgn4, tmidasgn5, tmidasgn6, tmidasgn7, tmidasgn8, tmidasgn9;
        ostringstream  outs;

        // Read matrices from filescdcdd
        ifilename = filePath + "Aext_res.mtx";
        MPI_Barrier(World);
        double tAread1 = MPI_Wtime();
        A.ParallelReadMM(ifilename, base, maximum<double>());
        MPI_Barrier(World);
        double tAread2 = MPI_Wtime();
        float Aext_load = A.LoadImbalance();
        if(myrank == 0) printf("\nreading file A done in %f seconds and load imabalance of Aext_res is %f\n\n", tAread2-tAread1, Aext_load);
        fflush(stdout);
        A.PrintInfo();

        // Read vectors from files
        vec1_filename = filePath + "A_vec_asgn_seg1.txt";
        vec2_filename = filePath + "A_vec_asgn_seg2.txt";
        vec3_filename = filePath + "A_vec_asgn_seg3.txt";
        FullyDistVec<int, int64_t> vec1, vec2, vec3;
        vec1.ParallelRead(vec1_filename, 0, maximum<int>());
        vec2.ParallelRead(vec2_filename, 0, maximum<int>());
        vec3.ParallelRead(vec3_filename, 0, maximum<int>());
        SpParHelper::Print("reading assigning vector segments done\n\n");

        
        //make a copy of A
        Dist::MPI_DCCols A2(A);
        SpParHelper::Print("Made a copy of input matrix A \n\n");
        

        SpParHelper::Print("------------------- Running streaming update with integrated permute -------------------\n\n");
        
        SpParHelper::Print("\n-------             Assign 1             -------\n\n");
        ibfilename = filePath + "A_seg1.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        fflush(stdout);
        B.PrintInfo();
        
        // Call the matrix assignment function with the read data
        MPI_Barrier(World);
        tstasgn1 = MPI_Wtime();
        MatrixAssignmentWithPermute(A, B, vec1, vec1, pruneA, World);
        MPI_Barrier(World);
        tendasgn1 = MPI_Wtime();
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("Returned from first MatrixAssignmentWithPermute and load imabalance is %f\n", A_load);
        A.PrintInfo();
        outs << "1st asg: " << tendasgn1-tstasgn1 << '\n' << endl;
        SpParHelper::Print(outs.str());
        
        SpParHelper::Print("\n-------             Assign 2             -------\n\n");
        ibfilename = filePath + "A_seg2.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        B.PrintInfo();

        // Call the matrix assignment function with the read data
        MPI_Barrier(World);
        tstasgn2 = MPI_Wtime();
        MatrixAssignmentWithPermute(A, B, vec1, vec2, pruneA, World);
        MPI_Barrier(World);
        tendasgn2 = MPI_Wtime();
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("Returned from second MatrixAssignmentWithPermute and load imabalance is %f\n", A_load);

        A.PrintInfo();
        outs.str("");
        outs << "2nd asg: " << tendasgn2-tstasgn2 << '\n' << endl;
        SpParHelper::Print(outs.str());

        SpParHelper::Print("\n-------             Assign 3             -------\n\n");
        ibfilename = filePath + "A_seg3.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        B.PrintInfo();
        // Call the matrix assignment function with the read data
        MPI_Barrier(World);
        tstasgn3 = MPI_Wtime();
        MatrixAssignmentWithPermute(A, B, vec1, vec3, pruneA, World);
        MPI_Barrier(World);
        tendasgn3 = MPI_Wtime();

        A_load = A.LoadImbalance();
        if(myrank == 0) printf("Returned from third MatrixAssignmentWithPermute and load imabalance is %f\n", A_load);
        A.PrintInfo();
        outs.str("");
        outs << "3rd asg: " << tendasgn3-tstasgn3 << '\n' << endl;
        SpParHelper::Print(outs.str());

        SpParHelper::Print("\n-------             Assign 4             -------\n\n");
        ibfilename = filePath + "A_seg4.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        B.PrintInfo();
        // Call the matrix assignment function with the read data
        MPI_Barrier(World);
        tstasgn4 = MPI_Wtime();
        MatrixAssignmentWithPermute(A, B, vec2, vec1, pruneA, World);
        MPI_Barrier(World);
        tendasgn4 = MPI_Wtime();

        
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("Returned from fourth MatrixAssignmentWithPermute and load imabalance is %f\n", A_load);
        A.PrintInfo();
        outs.str("");
        outs << "4th asg: " << tendasgn4-tstasgn4 << '\n' << endl;
        SpParHelper::Print(outs.str());

        
        SpParHelper::Print("\n-------             Assign 5             -------\n\n");
        ibfilename = filePath + "A_seg5.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        B.PrintInfo();
        // Call the matrix assignment function with the read data
        MPI_Barrier(World);
        tstasgn5 = MPI_Wtime();
        MatrixAssignmentWithPermute(A, B, vec2, vec2, pruneA, World);
        MPI_Barrier(World);
        tendasgn5 = MPI_Wtime();

        
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("Returned from fifth MatrixAssignmentWithPermute and load imabalance is %f\n", A_load);
        A.PrintInfo();
        outs.str("");
        outs << "5th asg: " << tendasgn5-tstasgn5 << '\n' << endl;
        SpParHelper::Print(outs.str());
        
        
        SpParHelper::Print("\n-------             Assign 6             -------\n\n");
        ibfilename = filePath + "A_seg6.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        B.PrintInfo();
        // Call the matrix assignment function with the read data
        MPI_Barrier(World);
        tstasgn6 = MPI_Wtime();
        MatrixAssignmentWithPermute(A, B, vec2, vec3, pruneA, World);
        MPI_Barrier(World);
        tendasgn6 = MPI_Wtime();

        
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("Returned from sixth MatrixAssignmentWithPermute and load imabalance is %f\n", A_load);
        A.PrintInfo();
        outs.str("");
        outs << "6th asg: " << tendasgn6-tstasgn6 << '\n' << endl;
        SpParHelper::Print(outs.str());
        
        SpParHelper::Print("\n-------             Assign 7             -------\n\n");
        ibfilename = filePath + "A_seg7.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        B.PrintInfo();
        // Call the matrix assignment function with the read data
        MPI_Barrier(World);
        tstasgn7 = MPI_Wtime();
        MatrixAssignmentWithPermute(A, B, vec3, vec1, pruneA, World);
        MPI_Barrier(World);
        tendasgn7 = MPI_Wtime();

        
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("Returned from seventh MatrixAssignmentWithPermute and load imabalance is %f\n", A_load);
        A.PrintInfo();
        outs.str("");
        outs << "7th asg: " << tendasgn7-tstasgn7 << '\n' << endl;
        SpParHelper::Print(outs.str());

        SpParHelper::Print("\n-------             Assign 8             -------\n\n");
        ibfilename = filePath + "A_seg8.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        B.PrintInfo();
        // Call the matrix assignment function with the read data
        MPI_Barrier(World);
        tstasgn8 = MPI_Wtime();
        MatrixAssignmentWithPermute(A, B, vec3, vec2, pruneA, World);
        MPI_Barrier(World);
        tendasgn8 = MPI_Wtime();

        
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("Returned from eighth MatrixAssignmentWithPermute and load imabalance is %f\n", A_load);
        A.PrintInfo();
        outs.str("");
        outs << "8th asg: " << tendasgn8-tstasgn8 << '\n' << endl;
        SpParHelper::Print(outs.str());

        SpParHelper::Print("\n-------             Assign 9             -------\n\n");
        ibfilename = filePath + "A_seg9.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        B.PrintInfo();
        // Call the matrix assignment function with the read data
        MPI_Barrier(World);
        tstasgn9 = MPI_Wtime();
        MatrixAssignmentWithPermute(A, B, vec3, vec3, pruneA, World);
        MPI_Barrier(World);
        tendasgn9 = MPI_Wtime();
        
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("Returned from ninth MatrixAssignmentWithPermute and load imabalance is %f\n", A_load);
        A.PrintInfo();
        outs.str("");
        outs << "9th asg: " << tendasgn9-tstasgn9 << '\n' << endl;
        SpParHelper::Print(outs.str());

        
        
        outs.str("");
        outs << "Total asgn time: " << tendasgn9-tstasgn9+tendasgn8-tstasgn8+tendasgn7-tstasgn7+tendasgn6-tstasgn6+tendasgn5-tstasgn5+tendasgn4-tstasgn4+tendasgn3-tstasgn3+tendasgn2-tstasgn2+tendasgn1-tstasgn1 << endl;
        SpParHelper::Print(outs.str());
        outs.str("");
        outs << "1st asg: " << tendasgn1-tstasgn1 << " 2nd asg: " << tendasgn2-tstasgn2 << " 3rd asg: " <<  tendasgn3-tstasgn3 << " 4th asg: " <<  tendasgn4-tstasgn4 << endl;
        SpParHelper::Print(outs.str());
        outs.str("");
        outs << "5st asg: " << tendasgn5-tstasgn5 << " 6nd asg: " << tendasgn6-tstasgn6 << " 7rd asg: " <<  tendasgn7-tstasgn7 << " 8th asg: " <<  tendasgn8-tstasgn8 << " 9th asg: " <<  tendasgn9-tstasgn9 << endl;
        SpParHelper::Print(outs.str());
        
        SpParHelper::Print("------------------- Done running streaming update with integrated permute -------------------\n\n");
        
        A.FreeMemory();
        B.FreeMemory();
        A = A2;
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("A load imabalance before start of new streaming experiment is %f\n", A_load);
        fflush(stdout);
        A.PrintInfo();
        A2.FreeMemory();
        
        SpParHelper::Print("------------------- Running our assign with and without seperate permute -------------------\n\n\n");

        SpParHelper::Print("\n-------             Assign 1             -------\n\n");
        ibfilename = filePath + "A_seg1.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        fflush(stdout);
        B.PrintInfo();
        
        // --------- 1st ----- Call the matrix assignment function with the read data --------- 1st ----- 
        MPI_Barrier(World);
        tstasgn1 = MPI_Wtime();
        MatrixAssignment(A, B, vec1, vec1, pruneA, World);
        MPI_Barrier(World);
        tmidasgn1 = MPI_Wtime();
        B.FreeMemory();
        SpParHelper::Print("matrix assign done\n");
        A.PrintInfo();
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("\nReturned from first matrixAssignement and load imabalance is %f\n", A_load);

        // Make a copy of A after just assignment
        Dist::MPI_DCCols Anew(A);

        // --------- 1st ----- Permute the new matrix and time it - discard it --------- 1st ----- 
        MatrixPermute(Anew, World);
        MPI_Barrier(World);
        tendasgn1 = MPI_Wtime();
        Anew_load = Anew.LoadImbalance();
        if(myrank == 0) printf("\nReturned from first MatrixPermute after matrixAssignement and load imabalance is %f\n", Anew_load);
        Anew.PrintInfo();
        Anew.FreeMemory();

        SpParHelper::Print("\n-------             Assign 2             -------\n\n");
        ibfilename = filePath + "A_seg2.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        B.PrintInfo();

        // --------- 2nd -----  Call the matrix assignment function with the read data    --------- 2nd -----
        MPI_Barrier(World);
        tstasgn2 = MPI_Wtime();
        MatrixAssignment(A, B, vec1, vec2, pruneA, World);
        MPI_Barrier(World);
        tmidasgn2 = MPI_Wtime();
        B.FreeMemory();
        SpParHelper::Print("matrix assign done\n");
        A.PrintInfo();
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("\nReturned from second matrixAssignement and load imabalance is %f\n", A_load);

        // --------- 2nd ----- Permute the new matrix and time it - discard it --------- 2nd ----- 
        Anew = A;
        MatrixPermute(Anew, World);
        MPI_Barrier(World);
        tendasgn2 = MPI_Wtime();
        Anew_load = Anew.LoadImbalance();
        if(myrank == 0) printf("\nReturned from second MatrixPermute after matrixAssignement and load imabalance is %f\n", Anew_load);
        Anew.PrintInfo();
        Anew.FreeMemory();

        SpParHelper::Print("\n-------             Assign 3             -------\n\n");
        ibfilename = filePath + "A_seg3.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        B.PrintInfo();

        // Call the matrix assignment function with the read data
        MPI_Barrier(World);
        tstasgn3 = MPI_Wtime();
        MatrixAssignment(A, B, vec1, vec3, pruneA, World);
        MPI_Barrier(World);
        tmidasgn3 = MPI_Wtime();
        B.FreeMemory();
        SpParHelper::Print("matrix assign done\n");
        A.PrintInfo();
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("\nReturned from third matrixAssignement and load imabalance is %f\n", A_load);

        // --------- 3rd ----- Permute the new matrix and time it - discard it --------- 3rd ----- 
        Anew = A;
        MatrixPermute(Anew, World);
        MPI_Barrier(World);
        tendasgn3 = MPI_Wtime();
        Anew_load = Anew.LoadImbalance();
        if(myrank == 0) printf("\nReturned from third MatrixPermute after matrixAssignement and load imabalance is %f\n", Anew_load);
        Anew.PrintInfo();
        Anew.FreeMemory();

        SpParHelper::Print("\n-------             Assign 4             -------\n\n");
        ibfilename = filePath + "A_seg4.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        B.PrintInfo();

        // Call the matrix assignment function with the read data
        MPI_Barrier(World);
        tstasgn4 = MPI_Wtime();
        MatrixAssignment(A, B, vec2, vec1, pruneA, World);
        MPI_Barrier(World);
        tmidasgn4 = MPI_Wtime();
        B.FreeMemory();
        SpParHelper::Print("matrix assign done\n");
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("\nReturned from fourth matrixAssignement and load imabalance is %f\n", A_load);

        // --------- 4th ----- Permute the new matrix and time it - discard it --------- 4th ----- 
        Anew = A;
        MatrixPermute(Anew, World);
        MPI_Barrier(World);
        tendasgn4 = MPI_Wtime();
        Anew_load = Anew.LoadImbalance();
        if(myrank == 0) printf("\nReturned from fouth MatrixPermute after matrixAssignement and load imabalance is %f\n", Anew_load);
        Anew.PrintInfo();
        Anew.FreeMemory();
        
        SpParHelper::Print("\n-------             Assign 5             -------\n\n");
        ibfilename = filePath + "A_seg5.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        B.PrintInfo();
        // Call the matrix assignment function with the read data
        MPI_Barrier(World);
        tstasgn5 = MPI_Wtime();
        MatrixAssignment(A, B, vec2, vec2, pruneA, World);
        MPI_Barrier(World);
        tmidasgn5 = MPI_Wtime();
        B.FreeMemory();
        SpParHelper::Print("matrix assign done\n");
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("\nReturned from fifth matrixAssignement and load imabalance is %f\n", A_load);
        A.PrintInfo();

        // --------- 5th ----- Permute the new matrix and time it - discard it --------- 5th ----- 
        Anew = A;
        MatrixPermute(Anew, World);
        MPI_Barrier(World);
        tendasgn5 = MPI_Wtime();
        Anew_load = Anew.LoadImbalance();
        if(myrank == 0) printf("\nReturned from fifth MatrixPermute after matrixAssignement and load imabalance is %f\n", Anew_load);
        Anew.PrintInfo();
        Anew.FreeMemory();
        
    
        SpParHelper::Print("\n-------             Assign 6             -------\n\n");
        ibfilename = filePath + "A_seg6.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        B.PrintInfo();

        // Call the matrix assignment function with the read data
        MPI_Barrier(World);
        tstasgn6 = MPI_Wtime();
        MatrixAssignment(A, B, vec2, vec3, pruneA, World);
        MPI_Barrier(World);
        tmidasgn6 = MPI_Wtime();
        B.FreeMemory();
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("\nReturned from sixth matrixAssignement and load imabalance is %f\n", A_load);
        
         // --------- 6th ----- Permute the new matrix and time it - discard it --------- 6th ----- 
        Anew = A;
        MatrixPermute(Anew, World);
        MPI_Barrier(World);
        tendasgn6 = MPI_Wtime();
        Anew_load = Anew.LoadImbalance();
        if(myrank == 0) printf("\nReturned from sixth MatrixPermute after matrixAssignement and load imabalance is %f\n", Anew_load);
        Anew.PrintInfo();
        Anew.FreeMemory();

        SpParHelper::Print("\n-------             Assign 7             -------\n\n");
        ibfilename = filePath + "A_seg7.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        B.PrintInfo();

        // Call the matrix assignment function with the read data
        MPI_Barrier(World);
        tstasgn7 = MPI_Wtime();
        MatrixAssignment(A, B, vec3, vec1, pruneA, World);
        MPI_Barrier(World);
        tmidasgn7 = MPI_Wtime();
        B.FreeMemory();
        SpParHelper::Print("matrix assign done\n");
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("\nReturned from seventh matrixAssignement and load imabalance is %f\n", A_load);
        A.PrintInfo();

        // --------- 7th ----- Permute the new matrix and time it - discard it --------- 7th ----- 
        Anew = A;
        MatrixPermute(Anew, World);
        MPI_Barrier(World);
        tendasgn7 = MPI_Wtime();
        Anew_load = Anew.LoadImbalance();
        if(myrank == 0) printf("\nReturned from seventh MatrixPermute after matrixAssignement and load imabalance is %f\n", Anew_load);
        Anew.PrintInfo();
        Anew.FreeMemory();


        SpParHelper::Print("\n-------             Assign 8             -------\n\n");
        ibfilename = filePath + "A_seg8.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        B.PrintInfo();

        // Call the matrix assignment function with the read data
        MPI_Barrier(World);
        tstasgn8 = MPI_Wtime();
        MatrixAssignment(A, B, vec3, vec2, pruneA, World);
        MPI_Barrier(World);
        tmidasgn8 = MPI_Wtime();
        B.FreeMemory();
        SpParHelper::Print("matrix assign done\n");
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("\nReturned from eighth matrixAssignement and load imabalance is %f\n", A_load);
        A.PrintInfo();

        // --------- 8th ----- Permute the new matrix and time it - discard it --------- 8th ----- 
        Anew = A;
        MatrixPermute(Anew, World);
        MPI_Barrier(World);
        tendasgn8 = MPI_Wtime();
        Anew_load = Anew.LoadImbalance();
        if(myrank == 0) printf("\nReturned from eighth MatrixPermute after matrixAssignement and load imabalance is %f\n", Anew_load);
        Anew.PrintInfo();
        Anew.FreeMemory();


        SpParHelper::Print("\n-------             Assign 9             -------\n\n");
        ibfilename = filePath + "A_seg9.mtx";
        B.ParallelReadMM(ibfilename, base, maximum<double>());
        B_load = B.LoadImbalance();
        if(myrank == 0) printf("reading file B done and load imabalance is %f\n", B_load);
        B.PrintInfo();

        // Call the matrix assignment function with the read data
        MPI_Barrier(World);
        tstasgn9 = MPI_Wtime();
        MatrixAssignment(A, B, vec3, vec3, pruneA, World);
        MPI_Barrier(World);
        tmidasgn9 = MPI_Wtime();
        B.FreeMemory();
        SpParHelper::Print("matrix assign done\n");
        A_load = A.LoadImbalance();
        if(myrank == 0) printf("\nReturned from ninth matrixAssignement and load imabalance is %f\n", A_load);

        // --------- 8th ----- Permute the new matrix and time it - discard it --------- 8th ----- 
        Anew = A;
        MatrixPermute(Anew, World);
        MPI_Barrier(World);
        tendasgn9 = MPI_Wtime();
        Anew_load = Anew.LoadImbalance();
        if(myrank == 0) printf("\nReturned from ninth MatrixPermute after matrixAssignement and load imabalance is %f\n", Anew_load);
        Anew.PrintInfo();
        Anew.FreeMemory();

        // ostringstream  outs;
        outs.str("");
        outs << "Total asgn time if only assign: " << tmidasgn9-tstasgn9+tmidasgn8-tstasgn8+tmidasgn7-tstasgn7+tmidasgn6-tstasgn6+tmidasgn5-tstasgn5+tmidasgn4-tstasgn4+tmidasgn3-tstasgn3+tmidasgn2-tstasgn2+tmidasgn1-tstasgn1 << endl;
        outs << "Total asgn time if permute done after assign: " << tendasgn9-tstasgn9+tendasgn8-tstasgn8+tendasgn7-tstasgn7+tendasgn6-tstasgn6+tendasgn5-tstasgn5+tendasgn4-tstasgn4+tendasgn3-tstasgn3+tendasgn2-tstasgn2+tendasgn1-tstasgn1 << endl;
        outs << "1st asg: " << tmidasgn1-tstasgn1 << " 2nd asg: " << tmidasgn2-tstasgn2 << " 3rd asg: " <<  tmidasgn3-tstasgn3 << " 4th asg: " <<  tmidasgn4-tstasgn4 << endl;
        outs << "5st asg: " << tmidasgn5-tstasgn5 << " 6nd asg: " << tmidasgn6-tstasgn6 << " 7rd asg: " <<  tmidasgn7-tstasgn7 << " 8th asg: " <<  tmidasgn8-tstasgn8 << " 9th asg: " <<  tmidasgn9-tstasgn9 << endl;
        outs << "1st perm: " << tendasgn1-tmidasgn1 << " 2nd perm: " << tendasgn2-tmidasgn2 << " 3rd perm: " <<  tendasgn3-tmidasgn3 << " 4th perm: " <<  tendasgn4-tmidasgn4 << endl;
        outs << "5st perm: " << tendasgn5-tmidasgn5 << " 6nd perm: " << tendasgn6-tmidasgn6 << " 7rd perm: " <<  tendasgn7-tmidasgn7 << " 8th perm: " <<  tendasgn8-tmidasgn8 << " 9th perm: " <<  tendasgn9-tmidasgn9 << endl;
        SpParHelper::Print(outs.str());
        
        SpParHelper::Print("------------------- Done Running our assign with and without seperate permute  -------------------\n\n");
      
    }
    
    MPI_Finalize();
    return 0;
}

