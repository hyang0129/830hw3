#include<stdio.h>
#include<string.h>
#include<algorithm>
#include<queue>
#include<vector>
#include<iostream>
using namespace std;

const double d = 0.85;
int V, E, L, M;

std::vector<std::vector<int>> in_edges;
std::vector<int> out_degree;


static const int blockSize = 1;


__global__ void oneVertex(int i, 
	const int V, 
	const double d,
	const int next,
	const int current, 
	const int* flat_edges,
	const int* edge_starts,
	const int* arr_out_degree, 
	double* arr_pr) {

	int idx = threadIdx.x;
	int sum = 0;


	for (int j = idx + edge_starts[i];
		j < edge_starts[i + 1]; j += blockSize) {
		int v = flat_edges[j];
		sum += arr_pr[v + current * V] / arr_out_degree[v];

	}

	__shared__ int r[blockSize];
	r[idx] = sum;
	__syncthreads();
	for (int size = blockSize / 2; size > 0; size /= 2) { //uniform
		if (idx < size)
			r[idx] += r[idx + size];
		__syncthreads();
	}

	if (idx == 0) {

		arr_pr[0] = edge_starts[i];
		arr_pr[1] = edge_starts[i+1];

		arr_pr[i + next * V] = 1.0;

		arr_pr[i + next * V] = (1.0 - d) / V + d * r[0];
	}
	

	}


int main(int argc, char** argv) {
	FILE* fin = fopen(argv[1], "r");
	FILE* fout = fopen(argv[2], "w");
	fscanf(fin, "%d%d%d%d", &V, &E, &L, &M);
	in_edges.resize(V);
	out_degree = std::vector<int>(V, 0);

	int longest_in_edges = 0;
	for (int i = 0; i < E; ++i) {
		int u, v;
		fscanf(fin, "%d%d", &u, &v);
		in_edges[v].push_back(u);
		++out_degree[u];

		// compute longest edge
		if (in_edges[v].size() > longest_in_edges) {
			longest_in_edges = in_edges[v].size();
		}

	}



	std::vector<double> pr[2];
	pr[0].resize(V);
	pr[1].resize(V);
	int current = 0;
	for (int i = 0; i < V; ++i) {
		pr[current][i] = 1.0 / V;
	}

	//create array equivalents 

	int* flat_edges = (int*)malloc(E * sizeof(int));
	int* edge_starts = new int[V + 1];

	int pos = 0; 

	for (int i = 0; i < V; ++i) {
		
		edge_starts[i] = pos; 

		for (int j = 0; j < in_edges[i].size(); j++) {
			flat_edges[pos] = in_edges[i][j];
			++pos;



		}
	}

	edge_starts[V] = E; 

	int* arr_out_degree = new int[V];
	for (int i = 0; i < V; ++i) {
		arr_out_degree[i] = out_degree[i];
	}

	double* arr_pr  = new double[V*2];
	//double arr_pr[V * 2];
	for (int i = 0; i < V; ++i) {
		arr_pr[i+current*V] = 1.0 / V;
	}

	
	////cuda allocate PR 
	cudaMallocManaged(&flat_edges, E * sizeof(int));
	cudaMallocManaged(&edge_starts, (V + 1) * sizeof(int));
	cudaMallocManaged(&arr_out_degree, V * sizeof(int));
	cudaMallocManaged(&arr_pr, 2 * V * sizeof(double));


	for (int iter = 0; iter < M; ++iter) {
		int next = 1 - current;

		for (int i = 0; i < V; ++i) {
			//double sum = 0;

			//for (int j = edge_starts[i]; j < edge_starts[i + 1]; ++j) {
			//	int v = flat_edges[j];
			//	
			//	sum += arr_pr[v + current * V] / arr_out_degree[v];
			//	
			//}

			//arr_pr[i + next * V] = (1.0 - d) / V + d * sum;
			

			oneVertex << <1, blockSize >> > (
				i,
				V,
				d,
				next,
				current,
				flat_edges,
				edge_starts,
				arr_out_degree,
				arr_pr
				);

			cudaDeviceSynchronize();
			
		}
		current = next;
	}

	cudaDeviceSynchronize();

	for (int i = 0; i < V * 2; ++i) {
		cout << arr_pr[i];
		cout << endl;
	}
	for (int i = 0; i < V; ++i) {
		pr[current][i] = arr_pr[i + current * V];
		
	}

	for (int i = 0; i < V; ++i) {
		fprintf(fout, "%.8f\n", pr[current][i]);
	}

	cudaFree(flat_edges);
	cudaFree(edge_starts);
	cudaFree(arr_out_degree);
	cudaFree(arr_pr);

	fclose(fin);
	fclose(fout);

	return 0;
}

