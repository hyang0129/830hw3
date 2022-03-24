#include<stdio.h>
#include<string.h>
#include<algorithm>
#include<queue>
#include<vector>

using namespace std;

const double d = 0.85;
int V,E,L,M;

std::vector<std::vector<int>> in_edges;
std::vector<int> out_degree;


int main(int argc,char** argv){
	FILE* fin = fopen(argv[1],"r");
	FILE* fout = fopen(argv[2],"w");
	fscanf(fin,"%d%d%d%d",&V,&E,&L,&M);
	in_edges.resize(V);
	out_degree = std::vector<int>(V,0);

	int longest_in_edges = 0; 

	for(int i = 0;i < E;++i){
		int u,v;
		fscanf(fin,"%d%d",&u,&v);
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
	for(int i = 0;i < V;++i){
		pr[current][i] = 1.0 / V;
	}

	//create array equivalents 
	
	int** arr_in_edges; 
	arr_in_edges = new int* [V];
	for (int i = 0; i < V; ++i) {
		arr_in_edges[V] = new int[longest_in_edges]; 
		for (int j = 0; j < longest_in_edges; j++) {
			if (j < in_edges[i].size()) {
				arr_in_edges[i][j] = in_edges[i][j];
			}
			else {
				arr_in_edges[i][j] = -1; 
			}

		}
	}

	int* arr_out_degree;
	arr_out_degree = new int[V];
	for (int i = 0; i < V; ++i) {
		arr_out_degree[i] = out_degree[i];
	}

	double** arr_pr;
	arr_pr = new double* [2];
	arr_pr[0] = new double[v];
	arr_pr[1] = new double[v];

	int current = 0;
	for (int i = 0; i < V; ++i) {
		arr_pr[current][i] = 1.0 / V;
	}


	//cuda allocate PR 

	


	for(int iter = 0;iter < M;++iter){
		int next = 1 - current;
		for(int i = 0;i < V;++i){

			// parallelize this part first 
			/*double sum = 0;
			for(int j = 0;j < in_edges[i].size();++j){
				int v = in_edges[i][j];
				sum += pr[current][v] / out_degree[v];
			}*/

			double sum = 0;
			for (int j = 0; j < longest_in_edges; ++j) {
				int v = arr_in_edges[i][j];

				if (v > -1) {
					sum += arr_pr[current][v] / arr_out_degree[v];
				}
			}


			arr_pr[next][i] = (1.0 - d) / V + d * sum;
		}
		current = next;
	}

	for(int i = 0;i < V;++i){
		fprintf(fout,"%.8f\n", arr_pr[current][i]);
	}
	fclose(fin);
	fclose(fout);

	return 0;
}

