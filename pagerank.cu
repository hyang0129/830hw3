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


static const int blockSize = 1024;
static const int blocks = 1024*4;


__global__ void sum_sections(
	const int V,
	const int total_edge_sections,
	const int current,
	const int* flat_edges,
	const int* cu_edge_sections,
	const int* cu_edge_sections_to_vertex,
	const int* arr_out_degree,
	const double* arr_pr,
	double* sections_result
) {

	int idx = threadIdx.x;

	for (int section = blockIdx.x;
		section < total_edge_sections;
		section += blocks
		) {

		// for each section

		double sum = 0;
		int v = 0;

		for (int j = idx + cu_edge_sections[section];
			j < cu_edge_sections[section + 1]; 
			j += blockSize) {

			v = flat_edges[j];

			sum += arr_pr[v + current * V] / arr_out_degree[v];

		}

		__shared__ double r[blockSize];
		r[idx] = sum;
		__syncthreads();
		for (int size = blockSize / 2; size > 0; size /= 2) { //uniform
			if (idx < size)
				r[idx] += r[idx + size];
			__syncthreads();
		}

		if (idx == 0) {
			sections_result[section] = r[0];
			
		}

	}



}


__global__ void reduce_sections(
	const int V,
	const double d,
	const int next,
	const int* vertex_section_starts,
	double* arr_pr,
	const double* sections_result
	) {

	int idx = threadIdx.x;


	for (int vertexblock = blockIdx.x;
		vertexblock < V;
		vertexblock += blocks
		) {

		// for each vertexblock

		double sum = 0;

		for (int j = idx + vertex_section_starts[vertexblock];
			j < vertex_section_starts[vertexblock + 1]; 
			j += blockSize) {
			
			sum += sections_result[j];

		}

		__shared__ double r[blockSize];
		r[idx] = sum;
		__syncthreads();
		for (int size = blockSize / 2; size > 0; size /= 2) { //uniform
			if (idx < size)
				r[idx] += r[idx + size];
			__syncthreads();
		}

		if (idx == 0) {
			arr_pr[vertexblock + next * V] = (1.0 - d) / V + d * r[0];

		}

	}
}




__global__ void allVertex(
	const int V,
	const double d,
	const int next,
	const int current,
	const int* flat_edges,
	const int* edge_starts,
	const int* arr_out_degree,
	double* arr_pr
) {

	int idx = threadIdx.x;
	

	for (int vertexblock = blockIdx.x;
		vertexblock < V;
		vertexblock += blocks
		) {

		// for each vertexblock

		double sum = 0;
		int v = 0; 

		for (int j = idx + edge_starts[vertexblock];
			j < edge_starts[vertexblock + 1]; j += blockSize) {
			v = flat_edges[j];

			sum += arr_pr[v + current * V] / arr_out_degree[v];

		}

		__shared__ double r[blockSize];
		r[idx] = sum;
		__syncthreads();
		for (int size = blockSize / 2; size > 0; size /= 2) { //uniform
			if (idx < size)
				r[idx] += r[idx + size];
			__syncthreads();
		}

		if (idx == 0) {
			arr_pr[vertexblock + next * V] = (1.0 - d) / V + d * r[0];
		}

	}



}


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
	double sum = 0;
	int v = 0;

	for (int j = idx + edge_starts[i];
		j < edge_starts[i + 1]; j += blockSize) {
		v = flat_edges[j];

		sum += arr_pr[v + current * V] / arr_out_degree[v];

	}

	__shared__ double r[blockSize];
	r[idx] = sum;
	__syncthreads();
	for (int size = blockSize / 2; size > 0; size /= 2) { //uniform
		if (idx < size)
			r[idx] += r[idx + size];
		__syncthreads();
	}


	if (idx == 0) {
	
		arr_pr[i + next * V] = (1.0 - d) / V + d * r[0];
	}
	

	}


int main(int argc, char** argv) {
	FILE* fin = fopen(argv[1], "r");
	FILE* fout = fopen(argv[2], "w");
	fscanf(fin, "%d%d%d%d", &V, &E, &L, &M);
	in_edges.resize(V);
	out_degree = std::vector<int>(V, 0);

	for (int i = 0; i < E; ++i) {
		int u, v;
		fscanf(fin, "%d%d", &u, &v);
		in_edges[v].push_back(u);
		++out_degree[u];

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
	int* arr_out_degree = new int[V];
	double* arr_pr  = new double[V*2];
	//double arr_pr[V * 2];

	////cuda allocate PR 
	cudaMallocManaged(&flat_edges, E * sizeof(int));
	cudaMallocManaged(&edge_starts, (V + 1) * sizeof(int));
	cudaMallocManaged(&arr_out_degree, V * sizeof(int));
	cudaMallocManaged(&arr_pr, 2 * V * sizeof(double));


	//assign 
	int pos = 0;


	for (int i = 0; i < V; ++i) {

		edge_starts[i] = pos;


		for (int j = 0; j < in_edges[i].size(); j++) {
			flat_edges[pos] = in_edges[i][j];
			
			++pos;

		}

	}

	edge_starts[V] = E;


	for (int i = 0; i < V; ++i) {
		arr_out_degree[i] = out_degree[i];
	}

	for (int i = 0; i < V; ++i) {
		arr_pr[i + current * V] = 1.0 / V;
	}

	// standard
	for (int iter = 0; iter < M; ++iter) {
		int next = 1 - current;

		allVertex << <blocks, blockSize >> >(
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

		int same = 1;
		for (int i = 0; i < V; ++i) {
			if (arr_pr[i + current * V] != arr_pr[i + next * V]) {
				same = 0;
			}
		}

		if (same == 1) {
			break;
		}

		current = next;
	}



	cudaDeviceSynchronize();

	for (int i = 0; i < V * 2; ++i) {
		//cout << arr_pr[i];
		//cout << edge_starts[i];

		//cout << endl;
	}


	cout << endl;

	for (int i = 0; i < V; ++i) {
		pr[current][i] = arr_pr[i + current * V];

	/*	cout << arr_pr[i + current * V];
		cout << endl;*/
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


