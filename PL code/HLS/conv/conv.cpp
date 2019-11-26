#include <ap_int.h>
#include <iostream>
#include <math.h>
#include "hls_half.h"


void Conv_half1(ap_uint<1> relu_en,ap_uint<1> sigmoid,
		half fture_in[],half w[],half b[],ha;f feature_out[]
	)
{
#pragma HLS INTERFACE m_axi depth=4294967295 port=fture_in offset=slave
#pragma HLS INTERFACE m_axi depth=4294967295 port=feature_out offset=slave
#pragma HLS INTERFACE m_axi depth=4294967295 port=w offset=slave
#pragma HLS INTERFACE m_axi depth=4294967295 port=b offset=slave
#pragma HLS INTERFACE s_axilite port=relu_en
#pragma HLS INTERFACE s_axilite port=sigmoid
#pragma HLS INTERFACE s_axilite port=return
	#define	CHin  32
	#define	Hin  32
	#define	Win 32
	#define	CHout 32
	#define	Ky  3
	#define	Kx 3
	#define	Sy  1
	#define	Sx 1


	ap_uint<8> pad_x,pad_y;


half feature_in[Hin][Win][CHin];
half W[Ky][Kx][CHin][CHout];
half bias[CHout];
#pragma HLS ARRAY_PARTITION variable=W block factor=16 dim=3
#pragma HLS ARRAY_PARTITION variable=feature_in block factor=16 dim=3


loop:
{
#pragma HLS LOOP_MERGE
	for(int i=0;i<Hin;i++){
			for(int j=0;j<Win;j++){
				for(int k=0;k<CHin;k++){
	#pragma HLS PIPELINE
					feature_in[i][j][k] = fture_in[i*Win*CHin+j*CHin+k];
				}
			}
		}
		for(int i=0;i<Ky;i++){
				for(int j=0;j<Kx;j++){
					for(int k=0;k<CHin;k++){
						for(int l=0;l<CHout;l++){
	#pragma HLS PIPELINE
							W[i][j][k][l]=w[i*Kx*CHin*CHout+j*CHin*CHout+k*CHout+l];
						}
					}
				}
			}
		for(int i=0;i<CHout;i++){
	#pragma HLS PIPELINE
			bias[i]=b[i];
		}
}


	pad_x=(Kx-1)/2;pad_y=(Ky-1)/2;



	ap_uint<10> Wout=32;
	ap_uint<10> Hout=32;


	for(int cout=0;cout<CHout;cout++){

		for(int i=0;i<Hout;i++){

			for(int j=0;j<Wout;j++)
			{
#pragma HLS PIPELINE
				half sum=0;
				for(int ii=0;ii<Ky;ii++)
				{
					for(int jj=0;jj<Kx;jj++)
					{
						ap_int<10> h=i-pad_y+ii;
				        ap_int<10> w=j-pad_x+jj;
						if(h>=0 && w>=0 && h<Hin && w<Win)
						{
							for(int cin1=0;cin1<CHin;cin1++)
							{
								half tp1=feature_in[h][w][cin1]*W[ii][jj][cin1][cout];
								sum+=tp1;
							}
						}
					}
				}

				sum+=bias[cout];
				if(relu_en && sum<0)
					sum=0;
//				if(sigmoid)
//					sum = 1/(1+exp(-sum));
				feature_out[i*Wout*CHout+j*CHout+cout]=sum;
//				feature_out[i][j][cout]=sum;
			}
		}
	}

//	if(sigmoid){
//		for(int i=0;i<Hout;i++){
//				for(int j=0;j<Wout;j++){
//		#pragma HLS PIPELINE
//					float tp= feature_out[i*Wout*CHout+j*CHout];
//						feature_out[i*Wout*CHout+j*CHout] = 1/(1+exp(-tp));
//					}
//				}
//			}

			}
