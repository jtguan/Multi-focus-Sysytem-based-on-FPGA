#include <ap_int.h>
#include <iostream>
#include <hls_half.h>
#include <math.h>

void fusion(half img1[],half img2[],half decision[],half out[]){
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE m_axi depth=4294967295 port=img1 offset=slave
#pragma HLS INTERFACE m_axi depth=4294967295 port=img2 offset=slave
#pragma HLS INTERFACE m_axi depth=4294967295 port=decision offset=slave
#pragma HLS INTERFACE m_axi depth=4294967295 port=out offset=slave

#define H 64
#define W 64
#define CH 3
	half image1[H][W][CH];
	half image2[H][W][CH];
	half dec[H][W];
#pragma HLS ARRAY_PARTITION variable=image1 block factor=3 dim=3
#pragma HLS ARRAY_PARTITION variable=image2 block factor=3 dim=3
	for(int h=0;h<H;h++)
		for(int w=0;w<W;w++)
			for(int ch=0;ch<CH;ch++){
#pragma HLS PIPELINE
				image1[h][w][ch] = img1[h*W*CH+w*CH+ch];
			}
	for(int h=0;h<H;h++)
			for(int w=0;w<W;w++)
				for(int ch=0;ch<CH;ch++){
#pragma HLS PIPELINE
					image2[h][w][ch] = img2[h*W*CH+w*CH+ch];
				}
	for(int h=0;h<H/2;h++)
			for(int w=0;w<W/2;w++){
#pragma HLS PIPELINE
				half tp= 1/(1+exp(-decision[h*W/2+w]));
				dec[2*h][2*w] = tp;
				dec[2*h+1][2*w] = tp;
				dec[2*h][2*w+1] = tp;
				dec[2*h+1][2*w+1] = tp;
			}

	for(int h=0;h<H;h++)
			for(int w=0;w<W;w++)
				for(int ch=0;ch<CH;ch++){
	#pragma HLS PIPELINE
					out[h*W*CH+w*CH+ch] = image1[h][w][ch]*dec[h][w]+image2[h][w][ch]*(1-dec[h][w]);
				}

}
