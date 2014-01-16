################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/cudamerge.cu \
../src/merge_kernel.cu 

CU_DEPS += \
./src/cudamerge.d \
./src/merge_kernel.d 

OBJS += \
./src/cudamerge.o \
./src/merge_kernel.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -I/usr/local/cuda-5.5/samples/common/inc -G -g -O0 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc --device-c -G -I/usr/local/cuda-5.5/samples/common/inc -O0 -g -gencode arch=compute_30,code=compute_30 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


