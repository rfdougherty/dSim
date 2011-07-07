################################################################################
#
# Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#

# Note: This file is a very trimmed down version of the original
# 	Makefile from NVIDIA. (Bragi Sveinsson)

###############################################################
#
# Build script for project
#
###############################################################

# Source files
EXECUTABLE	:= diffusion


# Basic directory setup for SDK
# (override directories only if they are not already defined)
SRCDIR     ?= src
BINDIR     ?= bin
ROOTOBJDIR ?= obj
BINSUBDIR   := release

# Compiler-specific flags
NVCCFLAGS :=
CXXFLAGS  :=

ifeq ($(emu), 1)
	NVCCFLAGS   += -deviceemu
	CXXFLAGS		+= -D__DEVICE_EMULATION__
endif

################################################################################
# Set up target file
################################################################################
TARGETDIR := $(BINDIR)/$(BINSUBDIR)
TARGET    := $(TARGETDIR)/$(EXECUTABLE)

################################################################################
# Set up object files
################################################################################
OBJDIR := $(ROOTOBJDIR)/$(BINSUBDIR)


all: $(TARGET)

$(TARGET): $(OBJDIR)/diffusion.o $(OBJDIR)/spinSystem.o $(OBJDIR)/renderSpins.o $(OBJDIR)/shaders.o $(OBJDIR)/spinSystem.cu_o
	g++ -fPIC -o bin/release/diffusion $(OBJDIR)/diffusion.o $(OBJDIR)/spinSystem.o $(OBJDIR)/renderSpins.o $(OBJDIR)/shaders.o $(OBJDIR)/spinSystem.cu_o -L/usr/local/cuda/lib64 -L/usr/local/cuda/sdk/C/lib -L/usr/local/cuda/sdk/C/common/lib -lconfig++ -lcutil -lglut -lcudart -lGL -lGLU -lGLEW -lparamgl

$(OBJDIR)/diffusion.o: $(SRCDIR)/diffusion.cpp
	g++ -fno-strict-aliasing $(CXXFLAGS) -I. -I/usr/local/cuda/include -I/usr/local/cuda/sdk/C/common/inc -I./jama/ -I./tnt/ -DUNIX -g -o $(OBJDIR)/diffusion.o -c $(SRCDIR)/diffusion.cpp

$(OBJDIR)/spinSystem.o: $(SRCDIR)/spinSystem.cpp
	g++ -fno-strict-aliasing $(CXXFLAGS) -I. -I/usr/local/cuda/include -I/usr/local/cuda/sdk/C/common/inc -DUNIX -g -o $(OBJDIR)/spinSystem.o -c $(SRCDIR)/spinSystem.cpp

$(OBJDIR)/renderSpins.o: $(SRCDIR)/renderSpins.cpp
	g++ -fno-strict-aliasing $(CXXFLAGS) -I. -I/usr/local/cuda/include -I/usr/local/cuda/sdk/C/common/inc -DUNIX -g -o $(OBJDIR)/renderSpins.o -c $(SRCDIR)/renderSpins.cpp

$(OBJDIR)/shaders.o: $(SRCDIR)/shaders.cpp
	g++ -fno-strict-aliasing $(CXXFLAGS) -I. -I/usr/local/cuda/include -I/usr/local/cuda/sdk/C/common/inc -DUNIX -g -o $(OBJDIR)/shaders.o -c $(SRCDIR)/shaders.cpp

$(OBJDIR)/spinSystem.cu_o: $(SRCDIR)/spinSystem.cu
	nvcc -o $(OBJDIR)/spinSystem.cu_o -c $(SRCDIR)/spinSystem.cu --ptxas-options="-v" --compiler-options -fno-strict-aliasing -arch=sm_11 $(NVCCFLAGS) -I. -I/usr/local/cuda/include -I/usr/local/cuda/sdk/C/common/inc -DUNIX -g

clean:
	rm -rf $(OBJDIR)/*o $(TARGET)


