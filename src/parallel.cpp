#include <mpi.h>
#include "ParallelExtension/MPI_Control.h"
#include <iostream>


using namespace std;

int main (int argc,char *argv[])
{
  MPI_Init (&argc, &argv);	/* starts MPI */
  {
  PARALLEL_EXT::mpi_parallel_ext para(MPI_COMM_WORLD,4);
  para.printInfo();
  }
  MPI_Finalize();
  return 0;
}