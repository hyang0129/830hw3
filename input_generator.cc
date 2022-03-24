#include<stdio.h>
#include<string.h>
#include<random>
#include<string>
#include<algorithm>
#include<set>

std::mt19937 generator(12356789);
std::uniform_int_distribution<int> dist(0,std::numeric_limits<int>::max());

void dump_func(int cas,int V,int E,int L,int M){
	printf("dumping case %d\n",cas);
	std::string file_name = "sample" + std::to_string(cas) + ".in";
	FILE* fp = fopen(file_name.c_str(),"w");
	fprintf(fp,"%d %d %d %d\n",V,E,L,M);
	std::set<std::pair<int,int>> visited;
	std::vector<int> degree(V,0);
	for(int i = 0;i < V;++i){
		fprintf(fp,"%d %d\n",i,(i + 1) % V);
		visited.insert(std::make_pair(i,(i + 1) % V));
		++degree[i];
	}
	for(int i = V;i < E;++i){
		while(1){
			int u = dist(generator) % V;
			int v = dist(generator) % V;
			if(u == v || degree[u] >= L || visited.count(std::make_pair(u,v)))
				continue;
			visited.insert(std::make_pair(u,v));
			++degree[u];
			fprintf(fp,"%d %d\n",u,v);
			break;
		}
	}
	fclose(fp);
}

int main(){
	int maxV = 100000;
	int maxD = 1000;
	int maxE = 2000000;
	int maxL = 63;
	int maxQ = 10000;
	int maxM = 10000;
	dump_func(1,10,20,maxL,2);
	dump_func(2,10,20,maxL,maxM);
	dump_func(3,maxE / 50,maxE,maxL,100);
	dump_func(4,maxE / 50,maxE,maxL,1000);
	dump_func(5,maxE / 50,maxE,maxL,maxM);
	dump_func(6,maxV,maxE,maxL,100);
	dump_func(7,maxV,maxE,maxL,1000);
	for(int cas = 8;cas <= 10;++cas){
		dump_func(cas,maxV,maxE,maxL,maxM);
	}
	return 0;
}

