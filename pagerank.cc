#include<stdio.h>
#include<string.h>
#include<algorithm>
#include<queue>
#include<vector>


const double d = 0.85;
int V, E, L, M;

std::vector<std::vector<int>> in_edges;
std::vector<int> out_degree;


int main(int argc, char** argv) {
	FILE* fin = fopen(argv[1], "r");
	FILE* fout = fopen(argv[2], "w");
	fscanf(fin, "%d%d%d%d", &V, &E, &L, &M);
	in_edges.resize(V);
	out_degree = std::vector<int>(V, 0);

	int longest_in_edges = 0;
	int num_edges = 0;
	for (int i = 0; i < E; ++i) {
		int u, v;
		fscanf(fin, "%d%d", &u, &v);
		in_edges[v].push_back(u);
		++out_degree[u];

		// compute longest edge
		if (in_edges[v].size() > longest_in_edges) {
			longest_in_edges = in_edges[v].size();
		}

		++num_edges;
	}

	std::vector<double> pr[2];
	pr[0].resize(V);
	pr[1].resize(V);
	int current = 0;
	for (int i = 0; i < V; ++i) {
		pr[current][i] = 1.0 / V;
	}

	//create array equivalents 

	//int flat_edges[num_edges];
	//int flat_edge_locations[V + 1];

	//int current_edge = 0;

	//for (int i = 0; i < V; ++i) {
	//	flat_edge_locations[i] = current_edge;

	//	for (int j = 0; j < in_edges[i].size(); j++) {

	//		flat_edges[current_edge] = in_edges[i][j];
	//		++current_edge;
	//	}
	//}
	//flat_edge_locations[V + 1] = num_edges;

	int** arr_in_edges = malloc(V * sizeof(int*));

	for (int i = 0; i < V; ++i) {
		arr_in_edges[V] = malloc(in_edges[i].size() * sizeof(int));

		for (int j = 0; j < in_edges[i].size(); j++) {
			arr_in_edges[i][j] = in_edges[i][j];

		}
	}




	int arr_out_degree[V];
	for (int i = 0; i < V; ++i) {
		arr_out_degree[i] = out_degree[i];
	}

	double arr_pr[2][V];

	for (int i = 0; i < V; ++i) {
		arr_pr[current][i] = 1.0 / V;
	}

	////cuda allocate PR 


	for (int iter = 0; iter < M; ++iter) {
		int next = 1 - current;
		for (int i = 0; i < V; ++i) {
			double sum = 0;
			for (int j = 0; j < arr_in_edges[i].size(); ++j) {
				int v = arr_in_edges[i][j];
				sum += arr_pr[current][v] / arr_out_degree[v];
			}

			//int flat_edge_start = flat_edge_locations[i];
			//int flat_edge_end = flat_edge_locations[i + 1];

			//for (int j = flat_edge_start; j < flat_edge_end; ++j) {
			//	int v = flat_edges[j];
			//	sum += arr_pr[current][v] / arr_out_degree[v];
			//}

			arr_pr[next][i] = (1.0 - d) / V + d * sum;
		}
		current = next;
	}

	for (int i = 0; i < V; ++i) {
		pr[current][i] = arr_pr[current][i];
	}

	for (int i = 0; i < V; ++i) {
		fprintf(fout, "%.8f\n", pr[current][i]);
	}
	fclose(fin);
	fclose(fout);

	return 0;
}

