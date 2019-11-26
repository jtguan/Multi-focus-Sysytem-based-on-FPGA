#include "ap_axi_sdata.h"
#include "ap_int.h"
#include "math.h"
//#include "hls_float.h"


typedef ap_axiu<32,1,1,1> AXI_VAL;

int Axi_Transfer(AXI_VAL* in_data,AXI_VAL* out_data,int value,int loop){
	int temp;
	temp = in_data->data;
	if(loop==1){
		out_data->data = temp;
	}
	else {
		out_data->data = value;
	}
	out_data->dest = in_data->dest;
	out_data->id = in_data->id;
	out_data->keep = in_data->keep;
	out_data->last = in_data->last;
	out_data->strb = in_data->strb;
	out_data->user = in_data->user;
	return temp;
}

void conv_transfer_1(AXI_VAL* in_data,AXI_VAL* out_data){
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE axis port=in_data
#pragma HLS INTERFACE axis port=out_data


	int temp;
	float precision;
	int param[3],relu,sigmoid,H_out,W_out;


	for(int idx=0;idx<3;idx++){
		param[idx] = Axi_Transfer(in_data,out_data,0,1);
	}

	relu = param[0];
	sigmoid =param[1];
	precision = param[2];
#define H_in 32
#define W_in 32
#define CHin 32
#define CHout 32
#define H_ker 3
#define W_ker 3
#define H_str 1
#define W_str 1

	float Input[H_in][W_in][CHin],Weights[H_ker][W_ker][CHin][CHout],Bias[CHout];
#pragma HLS ARRAY_PARTITION variable=Input block factor=16 dim=3
#pragma HLS ARRAY_PARTITION variable=Weights block factor=16 dim=3
	for(int i=0;i<H_in;i++){
		for(int j=0;j<W_in;j++){
			for(int k=0;k<CHin;k++){
#pragma HLS PIPELINE
				temp = Axi_Transfer(in_data,out_data,1,0);
				Input[i][j][k] = temp/precision;
				}
			}
		}



		for(int idx=0;idx<CHout;idx++){
#pragma HLS PIPELINE
			 temp= Axi_Transfer(in_data,out_data,2,0);
			 Bias[idx] = temp/precision;
		}

		for(int i=0;i<H_ker;i++){
				for(int j=0;j<W_ker;j++){
					for(int k=0;k<CHin;k++){
						for(int l=0;l<CHout;l++){
#pragma HLS PIPELINE
							 temp= Axi_Transfer(in_data,out_data,3,0);
							 Weights[i][j][k][l] = temp/precision;
						}
					}
				}
			}
	#define H_out 32
	#define W_out 32
	#define H_pad 1
	#define W_pad 1

		int total = H_out*W_out*CHout;
		Axi_Transfer(in_data,out_data,total,0);
		Axi_Transfer(in_data,out_data,H_out,0);
		Axi_Transfer(in_data,out_data,W_out,0);

		int feature_out[H_out][W_out][CHout];
		for(int i=0;i<H_out;i++){
			for(int j=0;j<W_out;j++){
				for(int cout=0;cout<CHout;cout++){
#pragma  HLS PIPELINE
					float sum=0;
					for(int ii=0;ii<H_ker;ii++)
						for(int jj=0;jj<W_ker;jj++)
						{

							ap_int<16> h=i*H_str-H_pad+ii;
							ap_int<16> w=j*W_str-W_pad+jj;
							if(h>=0 && w>=0 && h<H_in && w<W_in)
							{
								for(int cin=0;cin<CHin;cin++)
								{
									float tp=Input[h][w][cin]*Weights[ii][jj][cin][cout];
									sum+=tp;
								}
							}
						}

					sum+=Bias[cout];
					if(relu && sum<0)
						sum=0;
					feature_out[i][j][cout]=sum*precision;
				}
			}
	}

		for(int h_out=0;h_out<H_out;h_out++){
			for(int w_out=0;w_out<W_out;w_out++){
				for(int chout=0; chout<CHout;chout++){
					Axi_Transfer(in_data,out_data,feature_out[h_out][w_out][chout],0);
				}
			}
		}

}
