# Nolan Kyhl
# Fundamentals of Computing
# Lab 11 - Digit Recognition Neural Network - Makefile

CMP = gcc
MAIN = project
FUNC = projectfunc
EXEC = project

$(EXEC): $(FUNC).o $(MAIN).o 
	$(CMP) $(FUNC).o $(MAIN).o -lm -g -o $(EXEC)

$(FUNC).o: $(FUNC).c $(FUNC).h 
	$(CMP) -c $(FUNC).c -o $(FUNC).o 

$(MAIN).o: $(MAIN).c $(FUNC).h
	$(CMP) -c $(MAIN).c -o $(MAIN).o 

clean:
	rm $(FUNC).o $(MAIN).o
	rm $(EXEC)

