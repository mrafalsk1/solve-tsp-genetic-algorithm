all: compile
	
compile:
	clear
	@echo " "

	nvcc -o traveling_salesman traveling_salesman.cu -lcurand

	@echo "Compilaçao concluida"
	@echo " "

clean:
	rm -rf *.out
	rm -rf ?
	rm -rf ??

hoje:
	date


