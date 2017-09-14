 
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <sys/time.h>
#include "IndexIVFPQ.h"
#include "IndexFlat.h"
#include "index_io.h"
#include "AutoTune.h"
#include "Index.h"
#include "utils.h"

#include <fstream>

float * fvecs_read (const char *fname,
                    size_t *d_out, size_t *n_out)
{
    FILE *f = fopen(fname, "r");
    if(!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d; *n_out = n;
    float *x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for(size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int *ivecs_read(const char *fname, size_t *d_out, size_t *n_out)
{
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc,char *argv[])
{
	double t0 = elapsed();

    const char *index_key = "IVF4096,Flat";

    // these ones have better memory usage
    // const char *index_key = "Flat";
    // const char *index_key = "PQ32";
    // const char *index_key = "PCA80,Flat";
    // const char *index_key = "IVF4096,PQ8+16";
    // const char *index_key = "IVF4096,PQ32";
    // const char *index_key = "IMI2x8,PQ32";
    // const char *index_key = "IMI2x8,PQ8+16";
    // const char *index_key = "OPQ16_64,IMI2x8,PQ8+16";

    //faiss::Index * index;

    size_t d;

    {
        size_t M = atoi(argv[1]);
		size_t nbits_per_idx = atoi(argv[2]);
		size_t ksub = pow(2, nbits_per_idx);
		printf("M = %lu,  nbits_per_idx = %lu, k* = %lu\n", M, nbits_per_idx, ksub);
		//size_t M = 16;	
		//size_t nbits_per_idx = 10;
		
		float MSE = 0;
		int addnum = 0;
		printf ("[%.3f s] Loading train set\n", elapsed() - t0);

        size_t nt;
        float *xt = fvecs_read("sift1M/sift_learn.fvecs", &d, &nt);
		size_t dsub = d/M;
        //printf ("[%.3f s] Preparing index \"%s\" d=%ld\n",
        //        elapsed() - t0, index_key, d);
	
		faiss::IndexFlatL2 coarse_quantizer (d);
		int ncentroids = int (4 * sqrt (nt));
		
        //index = faiss::index_factory(d, index_key);
		//IndexIVFPQ (
        //    Index * quantizer, size_t d, size_t nlist,
        //    size_t M, size_t nbits_per_idx);
		//faiss::IndexIVFPQ index (&coarse_quantizer, d,
         //                    ncentroids, M, nbits_per_idx);	//m = 8, k* = 2^8
		
		//printf ("Index Param:\n d = %lu\n ncentroids = %d\n M = %lu\n k* = %f\n ", 
		//		d, ncentroids, M, pow(2, int(nbits_per_idx)));
        printf ("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);
		
		//index.train(nt, xt);
			
		faiss::IndexPQ pqindex (d, M, nbits_per_idx);
		pqindex.train(nt, xt);
		printf("[%.3f s] Training finish\n", elapsed() - t0);
		/*
		for (size_t m=0; m<M; m++)
		{
			float mindis = 1e20;
			int idxm = -1;
			const float * xsub = xt + m * dsub;
			fvec_L2sqr_ny (dist, xsub, pqindex.pq.get_centroids(m,0),dsub, ksub);
			
			for (size_t i=0; i<ksub; i++)
			{
				float tmpdis = dist[i];
				if (tmpdis < mindis)
				{
					mindis = tmpdis;
					idxm = i;
				}
			}
			
			MSE += mindis;
			
		}*/
	
		//First generate a distance table
		//Then calculate the min distance between each subvector and related centroids
		
		printf("number of centroids: %lu\n", pqindex.pq.centroids.size()/ksub);
		for (size_t num = 0; num < nt; num++)
		{
			for (size_t m=0; m<M; m++)
			{
				float *dis_tables = new float[ksub];
				faiss::pairwise_L2sqr(dsub, 
									  1, 
									  xt + (num*M+m)*dsub,
									  ksub, 
									  pqindex.pq.get_centroids(m,0), 
									  dis_tables);
				float mindis = 1e20;
				//int idxm = -1;
				if(num ==1 && m == 1)
				{
					/*printf("dis_table of num=1, m=1:\n");
					for(int i = 0; i<ksub; i++)
					{	
						printf("i = %d : %f\n", i, dis_tables[i]);
					}*/
					
				}
				for (int i=0; i<ksub; i++)
				{
					float tmpdis = dis_tables[i];
					if(tmpdis < mindis)
					{
						mindis = tmpdis;
						//idxm = i;
					}
				}
				
				//printf("mindis %lu = %7g \n", num*M+m, mindis);
				MSE += mindis;
				
				delete dis_tables;
			}
		}
		/*
		printf("centroids are: %f, %f, %f, %f, .....\n", 
			   pqindex.pq.centroids[0], 
			   pqindex.pq.centroids[1], 
			   pqindex.pq.centroids[2], 
			   pqindex.pq.centroids[3]);
		printf("xt is: %f, %f, %f, %f, .....\n", xt[0], xt[1], xt[2], xt[3]);
		*/
		float total = 0;
		for (size_t iii=0;iii<nt;iii++)
		{
			total += xt[iii];
		}
		total = total/nt;
		//printf("average value in each dimension is: %f\n", total);
	
		
	
	
	
	
	
	
	
		//size_t nx = nt * M;
		//float *dis_tables = new float[]
		//pqindex.pq.compute_distance_tables(nx, xt, )
		
		
	
	
		
		
		std::vector<faiss::Index::idx_t> labels (nt);
        std::vector<float>               distance (nt);
		//printf("initial labels and dis finish\n");
		/*@param x           input vectors to search, size n * d
     	* @param labels      output labels of the NNs, size n*k
     	* @param distances   output pairwise distances, size n*k
     	  input n = 1
		  input d = dsub
		  so x = dsub
		  input k = 1
		*/
	
	
		//float * slice = new float[nt * dsub];
		//float * Xjm = new float[dsub];
		//printf("initial Xjm finish\n");
		//printf("dsub = %lu, nt = %lu\n", dsub, nt);
		//index.search(nt, xt, 1, distance.data(), labels.data());
		/*
		for (int i = 0; i < nt; i++)
		{
			pqindex.add(nt,xt);	
		}
		*/
		
		//pqindex.search(nt, xt, 1, distance.data(), labels.data());
		
		
		/*
		for(int m = 0; m < M; m++)
		{
			for(int j = 0; j < nt; j++)
			{
				
				memcpy(Xjm, xt + j*d + m*dsub,
					   dsub * sizeof(float));
				index.search(1, Xjm, 1, &distance, &labels);
				//for(int i=0; i<dsub; i++)
				MSE += distance;
				addnum ++;
				
			}
		}*/
	
		MSE = MSE/nt;
		printf("MSE =  ");
		printf("%.3f\n", MSE);
	
		std::ofstream data("MSEdata.txt",std::ios::app);
		if(data.is_open())
		{
			data<<M<<" "<<nbits_per_idx<<" "<<MSE<<"\n";
			data.close();
		}
		
	
		
	
        
        delete [] xt;
	
    }


    
	
	return 0;
}

