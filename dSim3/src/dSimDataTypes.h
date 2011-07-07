
#define UINT8_MAX 255
#define UINT16_MAX 65535

typedef unsigned char uint8;

typedef unsigned short int uint16;

typedef struct _spinData
{
	//float x;
	//float y;
	//float z;
	
	//unsigned int seed0;
	//unsigned int seed1;

	float signalMagnitude;
	float signalPhase;

	uint8 compartmentType;
	uint16 insideFiber;
	
}spinData;

/*
typedef struct _int3
{
	int x;
	int y;
	int z;
	
}int3;

typedef struct _uint3
{
	uint x;
	uint y;
	uint z;
	
}uint3;*/
