CC = g++
CFLAGS = -O3 -march=native -Wall
CXXFLAGS += -std=c++17

SRCS = main.cpp metropolis.cpp heston.cpp pso.cpp heston_fft.cpp fractional_heston.cpp
OBJS = $(addprefix $(OUTPUT_DIR)/, $(SRCS:.cpp=.o))
OUTPUT_DIR = output

all: create_dir heston_project

create_dir:
	mkdir -p $(OUTPUT_DIR)

heston_project: $(OBJS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(OUTPUT_DIR)/heston_project $(OBJS)

run: all
	./$(OUTPUT_DIR)/heston_project

$(OUTPUT_DIR)/%.o: %.cpp
	$(CC) $(CFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OUTPUT_DIR)
