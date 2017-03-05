#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>

#include "readjpeg.h"

typedef struct {
  	float r;
  	float g;
  	float b;
} pixel_t;

double timestamp() {
  	struct timeval tv;
  	gettimeofday (&tv, 0);
  	return tv.tv_sec + 1e-6*tv.tv_usec;
}

double report_mflops(double t, int w, int h, double n) {
	if(n > 1) {n /= 2;}
	int flops_per_win = 3 * (2*(n+1)*(n+1) - 1);
	int windows = (w - n) * (h - n);
	return (double) (flops_per_win * windows) / t / 1e6;
}

void blur_frame(int width, int height, int* blur_radii, pixel_t *in, pixel_t *out, int nthr) {
	omp_set_num_threads(nthr);
	#pragma omp parallel for
	for(int y = 0; y < height; y++) {
 		for(int x = 0; x < width; x++) {
			pixel_t t;
	  		int idx = y * width + x;
	  		int r = blur_radii[idx];
	  		memset(&t,0,sizeof(t));


	  		for(int yy = y-r; yy < y+r; yy++) {
	      		if(yy < 0 || yy >= height)
					continue;				      
	      		for(int xx = x-r; xx < x+r; xx++) {
			  		if(xx < 0 || xx >= width)
		    			continue;
					t.r += in[yy*width + xx].r;
		  			t.g += in[yy*width + xx].g;
		  			t.b += in[yy*width + xx].b;
				}
	   		}
	  		/* scale output (normalize) */
	  		float scale = (float)((2*r + 1)*(2*r + 1));
		  	//printf("r=%d, scale = %f\n", r, scale);

	  		scale = 1.0f / scale;
	  		t.r *= scale;
	  		t.g *= scale;
	  		t.b *= scale;

		  	out[idx] = t;
		}
    }
}

void convert_to_pixel(pixel_t *out, frame_ptr in) {
  	for(int y = 0; y < in->image_height; y++) {
      	for(int x = 0; x < in->image_width; x++) {
	  		int r = (int)in->row_pointers[y][in->num_components*x + 0 ];
	  		int g = (int)in->row_pointers[y][in->num_components*x + 1 ];
	  		int b = (int)in->row_pointers[y][in->num_components*x + 2 ];
	  		out[y*in->image_width+x].r = (float)r;
	  		out[y*in->image_width+x].g = (float)g;
	  		out[y*in->image_width+x].b = (float)b;	 
		}
    }
}

void convert_to_frame(frame_ptr out, pixel_t *in) {
	for(int y = 0; y < out->image_height; y++) {
    	for(int x = 0; x < out->image_width; x++) {
	  		int r = (int)in[y*out->image_width + x].r;
	  		int g = (int)in[y*out->image_width + x].g;
	  		int b = (int)in[y*out->image_width + x].b;
	  		out->row_pointers[y][out->num_components*x + 0 ] = r;
	  		out->row_pointers[y][out->num_components*x + 1 ] = g;
	  		out->row_pointers[y][out->num_components*x + 2 ] = b;
		}
    }
}

int main(int argc, char *argv[]) {
  	int c;
  	char *inName = NULL;
  	char *outName = NULL;
  	int n=1;
  	int width=-1,height=-1;
  	frame_ptr frame;

  	pixel_t *inPix=NULL;
  	pixel_t *outPix=NULL;
  	int *blur_radii = NULL;

  	srand(5);

  	while((c = getopt(argc, argv, "i:n:o:"))!=-1) {
    	switch(c) {
			case 'i':
	  			inName = optarg;
	  		break;
			case 'o':
	  			outName = optarg;
	  		break;
			case 'n':
	  			n = atoi(optarg);
	  		break;
		}
    }

  	if(inName==0 || outName == 0) {
      	printf("need input filename and output filename\n");
      	return -1;
    }

  	frame = read_JPEG_file(inName);
  	if(!frame) {
      	printf("unable to read %s\n", inName);
      	exit(-1);
    }

  	width = frame->image_width;
  	height = frame->image_height;
 
  	inPix = new pixel_t[width*height];
  	outPix = new pixel_t[width*height];
  	blur_radii = new int[width*height];

  	for(int y = 0; y < height; y+=(height/16)) {
      	for(int x = 0; x < width; x+=(width/16)) {
	  		int r = 1;
	  		if(n > 1) {
	      		r += (rand() % n);
	    	}
	  		for(int yy = y; yy < std::min(height, (y + (height/16))); yy++) {
	      		for(int xx = x; xx < std::min(width, (x + (width/16))); xx++) {
		  			blur_radii[yy*width+xx] = r;
				}
	    	}
		}
    }
  
  	convert_to_pixel(inPix, frame);

	for(int i=1; i<=8; i++) {
  		double t0 = timestamp();
  		blur_frame(width, height, blur_radii, inPix, outPix, i);
  		t0 = timestamp() - t0;
  		printf("%i, %f \n", i, report_mflops(t0, width, height, 1.0));
  	}
  	
  	convert_to_frame(frame, outPix);
	write_JPEG_file(outName,frame,75);
  	destroy_frame(frame);

  	delete [] blur_radii;
  	delete [] inPix; 
  	delete [] outPix;
  	return 0;
}
