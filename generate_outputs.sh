

for i in {1..10}
do
	printf "\nworking on case ${i}:\n"
	test_data=sample${i}.in
	correct_file=sample${i}.out
	timeout 120s bash compile.sh
	CUDA_VISIBLE_DEVICES=8 time taskset -c 1-8 bash run.sh ${test_data} correct_file

done
